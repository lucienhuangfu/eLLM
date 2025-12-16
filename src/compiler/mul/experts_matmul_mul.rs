// === runner/experts_matmul_down.rs ===
#![allow(non_snake_case)]

use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::ExpertsDownTrait; // ← 新 trait

/// Experts Down Projection:
///   NONLIN[e, b, Hmid]   ×  W_down[e, Hmid, H]   → OUT[b, slot, H]
///
/// 其中：
/// - routing 使用 experts_indicator / indice_ptr / weight_ptr（expert-major）
/// - 写出 slot 使用 experts_topk_ptr（token-major, 每 token 升序 expert id 列表）
/// - 不做 residual
#[derive(Clone)]
pub struct ExpertsMatMulDown<T> {
    pub nonlin_ptr: ConstPtr<T>,   // [E, B, Hmid]
    pub wdown_nt_ptr: ConstPtr<T>, // [E, H, Hmid] (转置后)

    pub experts_indicator: ConstPtr<bool>, // [E]
    pub indice_ptr: ConstPtr<bool>,        // [E, B]
    pub weight_ptr: ConstPtr<T>,           // [E, B]

    // 每个 token 的 top-k expert 列表（升序 expert id）：
    // shape = [B, K]，K = num_topk
    pub experts_topk_ptr: ConstPtr<usize>,

    pub output_ptr: MutPtr<T>, // [B, K, H]，K = num_topk

    pub num_experts: usize,
    pub num_token: usize,
    pub hmid: usize,
    pub h: usize,
    pub num_topk: usize,

    pub params: MatMulParams,
    _marker: PhantomData<T>,

    // scratch
    pub b_panel_pool: Box<[T]>,
    pub b_panel_stride: usize,

    pub a_tile: Box<[T]>,   // MR × KC
    pub acc_tile: Box<[T]>, // MR × NR

    pub cpu_max_for_scratch: usize,
}

impl<T> ExpertsMatMulDown<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub unsafe fn new(
        nonlin_ptr: *const T,           // [E,B,Hmid]
        wdown_ptr: *const T,            // [E,Hmid,H]
        experts_indicator: *const bool, // [E]
        indice_ptr: *const bool,        // [E,B]
        weight_ptr: *const T,           // [E,B]
        experts_topk_ptr: *const usize, // [B,K] 每 token 的 top-k expert id
        output_ptr: *mut T,             // [B,K,H]

        num_experts: usize,
        num_token: usize,
        hmid: usize,
        h: usize,
        num_topk: usize,

        params: MatMulParams,
        cpu_max_for_scratch: usize,
    ) -> Self {
        let mb = params.a_row_step_macro.max(1);
        let nb = params.b_row_step_macro.max(1);
        let kc = params.column_step_macro.max(1);
        let mr = params.a_row_step_micro.max(1);
        let nr = params.b_row_step_micro.max(1);

        // -------- (1) 转置 W_down[e]: (Hmid×H) → (H×Hmid) --------
        let mut wdown_nt = vec![T::default(); num_experts * h * hmid];
        for e in 0..num_experts {
            let src = wdown_ptr.add(e * hmid * h);
            let dst = wdown_nt.as_mut_ptr().add(e * h * hmid);

            for kk in 0..hmid {
                let src_row = src.add(kk * h);
                for jj in 0..h {
                    *dst.add(jj * hmid + kk) = *src_row.add(jj);
                }
            }
        }

        // -------- (2) B_panel 池 --------
        let b_panel_stride = kc * nr;
        let b_panel_pool =
            vec![T::default(); cpu_max_for_scratch * b_panel_stride].into_boxed_slice();

        // -------- (3) A_tile / acc_tile --------
        let a_tile = vec![T::default(); mr * kc].into_boxed_slice();
        let acc_tile = vec![T::default(); mr * nr].into_boxed_slice();

        Self {
            nonlin_ptr: ConstPtr { ptr: nonlin_ptr },
            wdown_nt_ptr: ConstPtr {
                ptr: wdown_nt.as_ptr(),
            },

            experts_indicator: ConstPtr {
                ptr: experts_indicator,
            },
            indice_ptr: ConstPtr { ptr: indice_ptr },
            weight_ptr: ConstPtr { ptr: weight_ptr },

            experts_topk_ptr: ConstPtr {
                ptr: experts_topk_ptr,
            },

            output_ptr: MutPtr { ptr: output_ptr },

            num_experts,
            num_token,
            hmid,
            h,
            num_topk,
            params,
            _marker: PhantomData,

            b_panel_pool,
            b_panel_stride,

            a_tile,
            acc_tile,

            cpu_max_for_scratch,
        }
    }

    #[inline(always)]
    fn thread_b_panel(&self, tid: usize) -> *mut T {
        unsafe { self.b_panel_pool.as_ptr().add(tid * self.b_panel_stride) as *mut T }
    }

    /// slot(b,e)：在 experts_topk_ptr[b,*] 这一行里找到 expert e 的位置
    /// 形状：experts_topk_ptr: [B,K]，每行升序 expert id
    #[inline(always)]
    unsafe fn slot_of(&self, b: usize, e: usize) -> usize {
        let row = self.experts_topk_ptr.ptr.add(b * self.num_topk);
        for s in 0..self.num_topk {
            if *row.add(s) == e {
                return s;
            }
        }
        // 理论上不应走到这里（因为 e 对这个 b 激活时，一定在 topk 里）
        0
    }

    /// pack B: 从 NT 转置后的 W_down[e] 抽 KC×NR
    #[inline(always)]
    unsafe fn pack_b_panel(
        b_nt: *const T, // [H, Hmid]
        ldb_row: usize, // = Hmid
        n0: usize,
        k0: usize,
        kc: usize,
        nr: usize,
        out: *mut T,
    ) {
        for p in 0..kc {
            let col_k = k0 + p;
            let dst = out.add(p * nr);
            for lane in 0..nr {
                let j = n0 + lane;
                *dst.add(lane) = *b_nt.add(j * ldb_row + col_k);
            }
        }
    }

    /// pack A：把 be 行的 token 的 NONLIN[e][b][k0..k0+kc] 收到 a_tile
    #[inline(always)]
    unsafe fn pack_a_tile(
        &self,
        e: usize,
        k0: usize,
        be: usize,
        idx_buf: *const usize, // token 行号
        a_tile: *mut T,        // MR×KC
    ) {
        let mr = self.params.a_row_step_micro;
        let kc = self.params.column_step_macro;
        let base = self.nonlin_ptr.ptr.add(e * (self.num_token * self.hmid));

        // 有效行
        for r in 0..be {
            let b = *idx_buf.add(r);
            let src = base.add(b * self.hmid + k0);
            let dst = a_tile.add(r * kc);
            for p in 0..kc {
                *dst.add(p) = *src.add(p);
            }
        }
        // padding 行
        for r in be..mr {
            let dst = a_tile.add(r * kc);
            for p in 0..kc {
                *dst.add(p) = T::default();
            }
        }
    }

    pub fn run(
        &self,
                position_index: usize,
        position_interval: usize,
        batch_size: usize, // = num_token
        cpu_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let m = batch_size; // token
            let n = self.h; // hidden
            let k = self.hmid; // mid

            let mb = self.params.a_row_step_macro;
            let nb = self.params.b_row_step_macro;
            let kc = self.params.column_step_macro;
            let mr = self.params.a_row_step_micro;
            let nr = self.params.b_row_step_micro;

            let tiles_m = (m + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;

            let total_tiles = tiles_m * tiles_n;

            if let Some((tb, te)) = assign(total_tiles, cpu_num, thread_id) {
                let b_panel = self.thread_b_panel(thread_id);
                let a_tile = self.a_tile.as_ptr() as *mut T;
                let acc = self.acc_tile.as_ptr() as *mut T;

                // 临时 idx buffer（每线程一次分配）
                let mut idx_buf = vec![0usize; mb];

                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;

                    let slot0 = tm * mb;
                    let n0 = tn * nb;

                    let n_blk = (n - n0).min(nb);
                    // 这里默认 nb == nr（即 N 方向分块与微核宽度一致）
                    debug_assert!(n_blk <= nr);

                    // === 遍历 experts ===
                    for e in 0..self.num_experts {
                        if !*self.experts_indicator.ptr.add(e) {
                            continue;
                        }

                        // 该 expert 的 token 数 cnt_e
                        let mut cnt_e = 0usize;
                        for b in 0..batch_size {
                            if *self.indice_ptr.ptr.add(e * batch_size + b) {
                                cnt_e += 1;
                            }
                        }
                        if slot0 >= cnt_e {
                            continue;
                        }

                        let be = (cnt_e - slot0).min(mb);

                        // idx_buf 收集 slot0..slot0+be 的 token 行号
                        {
                            let mut s = 0usize;
                            let mut w = 0usize;
                            for b in 0..batch_size {
                                if *self.indice_ptr.ptr.add(e * batch_size + b) {
                                    if s >= slot0 && s < slot0 + be {
                                        idx_buf[w] = b;
                                        w += 1;
                                        if w == be {
                                            break;
                                        }
                                    }
                                    s += 1;
                                }
                            }
                        }

                        // 如果这个宏块内没有 token 命中，跳过
                        if be == 0 {
                            continue;
                        }

                        // === Kc×NR + MR×Kc → MR×NR ===
                        // 1) 先清零整个 acc（MR×NR），再在 K 维累加
                        for u in 0..(mr * nr) {
                            *acc.add(u) = T::default();
                        }

                        let mut k0 = 0usize;
                        while k0 < k {
                            // pack A (be 行)
                            self.pack_a_tile(e, k0, be, idx_buf.as_ptr(), a_tile);

                            // pack B（当前 expert 的 N×K 中的一段）
                            let b_nt = self.wdown_nt_ptr.ptr.add(e * (self.h * self.hmid));
                            Self::pack_b_panel(b_nt, self.hmid, n0, k0, kc, nr, b_panel);

                            // compute1: (MR×KC) * (KC×NR)，累加到 acc
                            self.compute1(a_tile as *const T, b_panel as *const T, acc);

                            k0 += kc;
                        }

                        // === finalize ×factor 并 scatter 写回 OUT[b,slot,H] ===
                        for r in 0..be {
                            let b = idx_buf[r];
                            let factor = *self.weight_ptr.ptr.add(e * batch_size + b);

                            // 用新的 experts_topk_ptr 求 slot(b,e)
                            let slot = self.slot_of(b, e);

                            let out_row = self
                                .output_ptr
                                .ptr
                                .add(b * (self.num_topk * n) + slot * n + n0);

                            // acc 中第 r 行的起点
                            let acc_row = acc.add(r * nr) as *const T;

                            // compute2: out_row += acc_row * factor
                            self.compute2(
                                out_row,
                                acc_row,
                                &factor as *const T,
                                n_blk, // 当前 tile 宽度
                            );
                        }
                    }
                }
            }
        }
    }
}

/* ---------------- ExpertsDownTrait 默认实现 ---------------- */

impl<T> ExpertsDownTrait<T> for ExpertsMatMulDown<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    // compute1: GEMM micro-kernel，占位（generic 不做事）
    default fn compute1(&self, _a_tile: *const T, _b_panel: *const T, _acc: *mut T) {
        // 默认空实现，真正的 f16 专用内核在下面特化
    }

    // compute2: out_row += acc_row * factor，占位（generic 不做事）
    default fn compute2(
        &self,
        _out_row: *mut T,
        _acc_row: *const T,
        _factor: *const T,
        _len: usize,
    ) {
        // 默认空实现
    }
}

/* ---------------- f16 专用实现（AVX-512 FP16） ---------------- */

impl ExpertsDownTrait<f16> for ExpertsMatMulDown<f16> {
    /// compute1: 用通用 3×32 GEMM 微核 matmul_block 累加到 acc
    fn compute1(&self, a_tile: *const f16, b_panel: *const f16, acc: *mut f16) {
        // 对 matmul_block 的参数映射：
        // - A_tile: MR×KC，行距 lda = KC
        // - B_panel: KC×NR，行距 32（微核内部固定）
        // - C(acc): MR×NR，行距 ldc = NR
        let kc = self.params.column_step_macro;
        let mr = self.params.a_row_step_micro;
        let nr = self.params.b_row_step_micro;

        debug_assert_eq!(mr, 3);
        debug_assert_eq!(nr, 32);

        let call_param = MatMulParams {
            a_row_step_macro: kc,  // lda = KC
            b_row_step_macro: nr,  // ldc = NR
            column_step_macro: kc, // kc
            a_row_step_micro: mr,  // MR
            b_row_step_micro: nr,  // NR
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::matmul_block::matmul_block(a_tile, b_panel, acc, &call_param);
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            unreachable!("avx512fp16 required for ExpertsDownTrait<f16>::compute1");
        }
    }

    /// compute2: out_row[j] += acc_row[j] * factor，长度 len
    fn compute2(&self, out_row: *mut f16, acc_row: *const f16, factor: *const f16, len: usize) {
        let factor_val = unsafe { *factor };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::moe_down::moe_down_scale_add(
                out_row, acc_row, factor_val, len,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            unreachable!("avx512fp16 required for ExpertsDownTrait<f16>::compute2");
        }
    }
}
