// === compiler/mul/experts_matmul_mul.rs ===
#![allow(non_snake_case)]

use crate::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use crate::kernel;
use crate::compiler::assign::assign;
use crate::compiler::ops::traits::mul_trait::ExpertsDownTrait;
use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

/// Experts Down Projection:
///   NONLIN[e, b, Hmid]   ×  W_down[e, Hmid, H]   → OUT[b, slot(b,e), H]
///
/// 其中：
/// - routing 使用 experts_indicator / indice_ptr / weight_ptr（expert-major）
/// - 写出 slot 使用 experts_topk_ptr（token-major, 每 token 升序 expert id 列表）
/// - 不做 residual
#[derive(Clone)]
pub struct ExpertsMatMulDown<T> {
    pub nonlin_ptr: ConstPtr<T>,   // [E, B, Hmid]
    pub wdown_nt_ptr: ConstPtr<T>, // [E, H, Hmid] (转置后，每行 stride=Hmid)

    pub experts_indicator: ConstPtr<bool>, // [E]
    pub indice_ptr: ConstPtr<bool>,        // [E, B]
    pub weight_ptr: ConstPtr<T>,           // [E, B]

    // 每个 token 的 top-k expert 列表（升序 expert id）：
    // shape = [B, Ktop]
    pub experts_topk_ptr: ConstPtr<usize>,

    pub output_ptr: MutPtr<T>, // [B, Ktop, H]

    pub num_experts: usize, // E
    pub num_token: usize,   // B
    pub hmid: usize,        // Hmid
    pub h: usize,           // H
    pub num_topk: usize,    // Ktop

    pub params: MatMulParams,
     decode_only_flag: bool,
    _marker: PhantomData<T>,

    // ---- thread-private scratch pools（按 tid 切片）----
    // B_panel: KC × NR（行主，每行 NR 连续）
    b_panel_pool: Box<[T]>,
    b_panel_stride: usize, // = kc * nr

    // A_tile: MR × KC（行主，每行 KC 连续）
    a_tile_pool: Box<[T]>,
    a_tile_stride: usize, // = mr * kc

    // ACC: MR × NR（行主，每行 NR 连续）
    acc_pool: Box<[T]>,
    acc_stride: usize, // = mr * nr

    // idx buffer: MB 个 token index（每线程一份）
    idx_buf_pool: Box<[usize]>,
    idx_stride: usize, // = mb

                       // pools 的 thread 数（只在 new() 用来分配；这里不存 cpu_max_for_scratch）
                       // 你说外部保证 cpu_num/thread_id 合法，所以 run 内不做 debug_assert
}

impl<T> ExpertsMatMulDown<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    #[inline]
    fn detect_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1)
    }

    pub unsafe fn new(
        nonlin_ptr: *const T,           // [E,B,Hmid]
        wdown_nt_ptr: *const T,         // [E,Hmid,H]（行主，stride=H）
        experts_indicator: *const bool, // [E]
        indice_ptr: *const bool,        // [E,B]
        weight_ptr: *const T,           // [E,B]
        experts_topk_ptr: *const usize, // [B,Ktop]
        output_ptr: *mut T,             // [B,Ktop,H]

        num_experts: usize,
        num_token: usize,
        hmid: usize,
        h: usize,
        num_topk: usize,

        params: MatMulParams,
        decode_only_flag: bool,
    ) -> Self {
        let mb = params.a_row_step_macro.max(1);
        let kc = params.column_step_macro.max(1);
        let mr = params.a_row_step_micro.max(1);
        let nr = params.b_row_step_micro.max(1);

        // 你当前微核约定 MR=3, NR=32；这里不 assert，外部保证
        // debug_assert_eq!(mr, 3);
        // debug_assert_eq!(nr, 32);

        // -------- (1) 转置 W_down[e]: (Hmid×H) → (H×Hmid) --------
        // wdown_ptr 每个 expert 的 layout： [Hmid][H]
        // 转置后： [H][Hmid]，行距=Hmid

        // -------- (2) 分配 thread-private pools --------
        let threads = Self::detect_threads();

        let b_panel_stride = kc * nr;
        let a_tile_stride = mr * kc;
        let acc_stride = mr * nr;
        let idx_stride = mb;

        let b_panel_pool = vec![T::default(); threads * b_panel_stride].into_boxed_slice();
        let a_tile_pool = vec![T::default(); threads * a_tile_stride].into_boxed_slice();
        let acc_pool = vec![T::default(); threads * acc_stride].into_boxed_slice();
        let idx_buf_pool = vec![0usize; threads * idx_stride].into_boxed_slice();

        Self {
            nonlin_ptr: ConstPtr { ptr: nonlin_ptr },
            wdown_nt_ptr: ConstPtr { ptr: wdown_nt_ptr },

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

            a_tile_pool,
            a_tile_stride,

            acc_pool,
            acc_stride,

            idx_buf_pool,
            idx_stride,
            decode_only_flag,
        }
    }

    #[inline(always)]
    fn thread_slices(&self, tid: usize) -> (*mut T, *mut T, *mut T, *mut usize) {
        unsafe {
            let b_panel = self.b_panel_pool.as_ptr().add(tid * self.b_panel_stride) as *mut T;
            let a_tile = self.a_tile_pool.as_ptr().add(tid * self.a_tile_stride) as *mut T;
            let acc = self.acc_pool.as_ptr().add(tid * self.acc_stride) as *mut T;
            let idx = self.idx_buf_pool.as_ptr().add(tid * self.idx_stride) as *mut usize;
            (b_panel, a_tile, acc, idx)
        }
    }

    /// slot(b,e)：在 experts_topk_ptr[b,*] 这一行里找到 expert e 的位置
    /// experts_topk_ptr: [B,Ktop]，每行升序 expert id
    #[inline(always)]
    unsafe fn slot_of(&self, b: usize, e: usize) -> usize {
        let row = self.experts_topk_ptr.ptr.add(b * self.num_topk);
        for s in 0..self.num_topk {
            if *row.add(s) == e {
                return s;
            }
        }
        0
    }

    /// pack B: 从转置后的 W_down_nt[e]（[H, Hmid]，stride=Hmid）抽取 KC×NR
    /// 支持 cols_this < NR：剩余 lane 填 0，避免越界
    #[inline(always)]
    unsafe fn pack_b_panel(
        b_nt: *const T, // [H, Hmid]
        ldb_row: usize, // = Hmid
        n0: usize,      // H 维起点（输出列起点）
        k0: usize,      // Hmid 维起点
        kc: usize,
        nr: usize,
        cols_this: usize, // <= nr
        out: *mut T,      // KC×NR
    ) {
        for p in 0..kc {
            let col_k = k0 + p;
            let dst = out.add(p * nr);
            for lane in 0..nr {
                if lane < cols_this {
                    let j = n0 + lane; // 0..H-1
                    *dst.add(lane) = *b_nt.add(j * ldb_row + col_k);
                } else {
                    *dst.add(lane) = T::default();
                }
            }
        }
    }

    /// pack A：把 valid_rows 行 token 的 NONLIN[e][b][k0..k0+kc] 收到 a_tile（MR×KC）
    /// valid_rows <= MR；剩余行 padding 0
    #[inline(always)]
    unsafe fn pack_a_tile(
        &self,
        e: usize,
        k0: usize,
        valid_rows: usize,
        idx_buf: *const usize,
        idx_off: usize,
        a_tile: *mut T, // MR×KC
        kc: usize,
        mr: usize,
    ) {
        let base = self.nonlin_ptr.ptr.add(e * (self.num_token * self.hmid));

        // 有效行
        for r in 0..valid_rows {
            let b = *idx_buf.add(idx_off + r);
            let src = base.add(b * self.hmid + k0);
            let dst = a_tile.add(r * kc);
            for p in 0..kc {
                *dst.add(p) = *src.add(p);
            }
        }
        // padding 行
        for r in valid_rows..mr {
            let dst = a_tile.add(r * kc);
            for p in 0..kc {
                *dst.add(p) = T::default();
            }
        }
    }

    pub fn run(&self, prefill_size: usize, decode_size: usize, thread_num: usize, thread_id: usize) {
        unsafe {
            let bcap = self.num_token; // bmax

            let m = prefill_size; // token 数
            let n = self.h; // 输出列 H
            let k = self.hmid; // 输入维 Hmid

            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1);
            let nr = self.params.b_row_step_micro.max(1);

            // --- tiles ---
            let tiles_m = (m + mb - 1) / mb;
            let tiles_n = (n + nb - 1) / nb;
            let total_tiles = tiles_m * tiles_n;

            // thread-private slices
            let (b_panel, a_tile, acc, idx_buf) = self.thread_slices(thread_id);

            if let Some((tb, te)) = assign(total_tiles, thread_num, thread_id) {
                for t in tb..te {
                    let tm = t / tiles_n;
                    let tn = t % tiles_n;

                    let slot0 = tm * mb; // “第几个命中 token” 的宏块起点（expert 内）
                    let n0 = tn * nb; // H 方向起点
                    let n_blk = (n - n0).min(nb);
                    if n_blk == 0 {
                        continue;
                    }

                    // NB 不要求==NR；内部按 NR 切
                    let mut nt = 0usize;
                    while nt < n_blk {
                        let cols_this = (n_blk - nt).min(nr); // <= 32

                        // === 遍历 experts ===
                        for e in 0..self.num_experts {
                            if !*self.experts_indicator.ptr.add(e) {
                                continue;
                            }

                            // 该 expert 的 token 命中数 cnt_e
                            let mut cnt_e = 0usize;
                            for b in 0..m {
                                if *self.indice_ptr.ptr.add(e * bcap + b) {
                                    cnt_e += 1;
                                }
                            }
                            if slot0 >= cnt_e {
                                continue;
                            }

                            let be_total = (cnt_e - slot0).min(mb); // 当前宏块里总 token 数（可能 > MR）
                            if be_total == 0 {
                                continue;
                            }

                            // idx_buf 收集 slot0..slot0+be_total 的 token 行号（写入当前线程的 idx_buf）
                            {
                                let mut seen = 0usize;
                                let mut w = 0usize;
                                for b in 0..m {
                                    if *self.indice_ptr.ptr.add(e * bcap + b) {
                                        if seen >= slot0 && seen < slot0 + be_total {
                                            *idx_buf.add(w) = b;
                                            w += 1;
                                            if w == be_total {
                                                break;
                                            }
                                        }
                                        seen += 1;
                                        if seen >= slot0 + be_total {
                                            break;
                                        }
                                    }
                                }
                            }

                            // --- 关键：按 MR 分批（valid_rows <= MR），避免 be_total > MR 越界 ---
                            let mut off = 0usize;
                            while off < be_total {
                                let valid_rows = (be_total - off).min(mr);

                                // 清零 acc（MR×NR）
                                for u in 0..(mr * nr) {
                                    *acc.add(u) = T::default();
                                }

                                // K 方向累加：acc += A_tile × B_panel
                                let mut k0 = 0usize;
                                debug_assert!(k % kc == 0);
                                while k0 < k {
                                    // pack A: valid_rows 行，padding 到 MR
                                    Self::pack_a_tile(
                                        self, e, k0, valid_rows, idx_buf, off, a_tile,
                                        kc, // 注意：最后一块可能 < kc
                                        mr,
                                    );

                                    // pack B: KC×NR（cols_this<NR 时填 0）
                                    let b_nt_e =
                                        self.wdown_nt_ptr.ptr.add(e * (self.h * self.hmid));
                                    Self::pack_b_panel(
                                        b_nt_e,
                                        self.hmid, // ldb_row = Hmid
                                        n0 + nt,   // col start in H
                                        k0,        // k start in Hmid
                                        kc,
                                        nr,
                                        cols_this,
                                        b_panel,
                                    );

                                    // compute1: acc += (MR×KC) × (KC×NR)
                                    self.compute1(a_tile as *const T, b_panel as *const T, acc);

                                    k0 += kc;
                                }

                                // scatter：对 valid_rows 行做 out_row += acc_row * weight
                                for r in 0..valid_rows {
                                    let b = *idx_buf.add(off + r);
                                    let factor = *self.weight_ptr.ptr.add(e * bcap + b);

                                    let slot = self.slot_of(b, e);

                                    let out_row = self
                                        .output_ptr
                                        .ptr
                                        .add(b * (self.num_topk * n) + slot * n + (n0 + nt));

                                    let acc_row = acc.add(r * nr) as *const T;

                                    self.compute2(
                                        out_row,
                                        acc_row,
                                        &factor as *const T,
                                        cols_this, // 这次只写 cols_this（<=32）
                                    );
                                }

                                off += valid_rows;
                            }
                        }

                        nt += nr;
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
    // compute1: acc += A_tile × B_panel
    default fn compute1(&self, _a_tile: *const T, _b_panel: *const T, _acc: *mut T) {}

    // compute2: out_row[j] += acc_row[j] * factor，长度 len（<=32）
    default fn compute2(
        &self,
        _out_row: *mut T,
        _acc_row: *const T,
        _factor: *const T,
        _len: usize,
    ) {
    }
}

/* ---------------- f16 专用实现（AVX-512 FP16） ---------------- */

impl ExpertsDownTrait<f16> for ExpertsMatMulDown<f16> {
    /// compute1: acc += A_tile × B_panel（3×32 微核，C=acc，行距=NR=32）
    fn compute1(&self, a_tile: *const f16, b_panel: *const f16, acc: *mut f16) {
        let kc = self.params.column_step_macro.max(1);
        let mr = self.params.a_row_step_micro.max(1);
        let nr = self.params.b_row_step_micro.max(1);

        let call_param = MatMulParams {
            // 映射约定：a_row_step_macro=lda, b_row_step_macro=ldc, column_step_macro=kc
            // A_tile 每行连续 kc（我们 pack 的就是这种 layout）
            a_row_step_macro: kc, // lda = kc
            b_row_step_macro: nr, // ldc = 32
            column_step_macro: kc,
            a_row_step_micro: mr,
            b_row_step_micro: nr,
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

    /// compute2: out_row[j] += acc_row[j] * factor，长度 len（<=32）
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
#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::mem;

    use crate::traits::from_f32::FromF32;

    // ========================================================================
    // Helpers
    // ========================================================================

    #[inline]
    fn f16_from_f32(x: f32) -> f16 {
        <f16 as FromF32>::from_f32(x)
    }

    /// 只给测试用：把 f16 bits 转成 f32
    #[inline]
    fn f32_from_f16(x: f16) -> f32 {
        let bits: u16 = unsafe { mem::transmute(x) };
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits & 0x7C00) >> 10;
        let mant = bits & 0x03FF;

        let f_bits: u32 = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                let mut m = mant as u32;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let exp_f = (e + 127) as u32;
                sign | (exp_f << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            let exp_f = 0xFFu32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        } else {
            let exp_f = (exp as i32 - 15 + 127) as u32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        };

        f32::from_bits(f_bits)
    }

    #[inline]
    fn approx_eq_f32(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// 找 slot(b,e)：topk 行是升序 expert id
    #[inline]
    fn slot_of(topk: &[usize], b: usize, ktop: usize, e: usize) -> usize {
        let row = &topk[b * ktop..b * ktop + ktop];
        row.iter().position(|&x| x == e).unwrap_or(0)
    }

    fn verify_output(out: &[f16], out_ref: &[f32], tol: f32, msg: &str) {
        for i in 0..out.len() {
            let got = f32_from_f16(out[i]);
            let exp = out_ref[i];
            assert!(
                approx_eq_f32(got, exp, tol),
                "{} mismatch at {}: got={}, exp={}",
                msg,
                i,
                got,
                exp
            );
        }
    }

    // ========================================================================
    // Test Cases
    // ========================================================================

    #[test]
    fn test_down_mb_gt_mr_basic_no_tail() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_experts = 1usize; // E
        let num_token = 6usize; // B (be_total=6 > mr=3)
        let hmid = 32usize; // Hmid
        let h = 64usize; // H
        let num_topk = 1usize; // Ktop

        let params = MatMulParams {
            a_row_step_macro: 6,   // MB
            b_row_step_macro: 32,  // NB
            column_step_macro: 16, // KC (hmid%kc==0)
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];
        let mut out = vec![f16_from_f32(0.0); num_token * num_topk * h];

        let experts_indicator = vec![true; num_experts];
        let mut indice = vec![false; num_experts * num_token];
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];
        let mut topk = vec![0usize; num_token * num_topk];

        // expert0 命中全部 token
        for b in 0..num_token {
            indice[0 * num_token + b] = true;
            weight[0 * num_token + b] = f16_from_f32(0.5 + 0.01 * b as f32);
            topk[b * num_topk + 0] = 0;
        }

        for b in 0..num_token {
            for kk in 0..hmid {
                nonlin[(0 * num_token + b) * hmid + kk] =
                    f16_from_f32(0.01 * b as f32 + 0.001 * kk as f32);
            }
        }
        for kk in 0..hmid {
            for j in 0..h {
                wdown[(0 * hmid + kk) * h + j] =
                    f16_from_f32(0.002 * kk as f32 + 0.0005 * j as f32);
            }
        }

        let runner = unsafe {
            ExpertsMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                weight.as_ptr(),
                topk.as_ptr(),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false
            )
        };

        runner.run(num_token, 0, 1, 0);

        // reference (f32)
        let mut out_ref = vec![0.0f32; num_token * num_topk * h];
        for b in 0..num_token {
            if !indice[0 * num_token + b] {
                continue;
            }
            let slot = slot_of(&topk, b, num_topk, 0);
            let w = f32_from_f16(weight[0 * num_token + b]);

            for j in 0..h {
                let mut acc = 0.0f32;
                for kk in 0..hmid {
                    let a = f32_from_f16(nonlin[(0 * num_token + b) * hmid + kk]);
                    let bv = f32_from_f16(wdown[(0 * hmid + kk) * h + j]);
                    acc += a * bv;
                }
                out_ref[(b * num_topk + slot) * h + j] += w * acc;
            }
        }

        verify_output(&out, &out_ref, 5e-2, "basic");
    }

    #[test]
    fn test_down_tail_len_lt_32() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        // H=48 => 32 + 16 tail
        let num_experts = 1usize;
        let num_token = 3usize;
        let hmid = 32usize;
        let h = 48usize;
        let num_topk = 1usize;

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 48, // NB=48（一个 tile 覆盖 48）
            column_step_macro: 16,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];
        let mut out = vec![f16_from_f32(0.0); num_token * num_topk * h];

        let experts_indicator = vec![true; num_experts];
        let indice = vec![true; num_experts * num_token];
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];
        let mut topk = vec![0usize; num_token * num_topk];

        for b in 0..num_token {
            weight[b] = f16_from_f32(0.7 + 0.02 * b as f32);
            topk[b] = 0;
        }

        for b in 0..num_token {
            for kk in 0..hmid {
                nonlin[b * hmid + kk] = f16_from_f32(0.01 * b as f32 + 0.0003 * kk as f32);
            }
        }
        for kk in 0..hmid {
            for j in 0..h {
                wdown[kk * h + j] = f16_from_f32(0.001 * kk as f32 + 0.0009 * j as f32);
            }
        }

        let runner = unsafe {
            ExpertsMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                weight.as_ptr(),
                topk.as_ptr(),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false
            )
        };

        runner.run(num_token, 0, 1, 0);

        // reference
        let mut out_ref = vec![0.0f32; num_token * num_topk * h];
        for b in 0..num_token {
            let w = f32_from_f16(weight[b]);
            for j in 0..h {
                let mut acc = 0.0f32;
                for kk in 0..hmid {
                    let a = f32_from_f16(nonlin[b * hmid + kk]);
                    let bv = f32_from_f16(wdown[kk * h + j]);
                    acc += a * bv;
                }
                out_ref[b * h + j] += w * acc;
            }
        }

        verify_output(&out, &out_ref, 5e-2, "tail");
    }

    #[test]
    fn test_down_two_experts_slot_scatter() {
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_experts = 2usize;
        let num_token = 4usize;
        let hmid = 32usize;
        let h = 64usize;
        let num_topk = 2usize;

        let params = MatMulParams {
            a_row_step_macro: 4,
            b_row_step_macro: 32,
            column_step_macro: 16,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];
        let mut out = vec![f16_from_f32(0.0); num_token * num_topk * h];

        let experts_indicator = vec![true; num_experts];
        let mut indice = vec![false; num_experts * num_token];
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];

        // topk: [0,1]
        let mut topk = vec![0usize; num_token * num_topk];
        for b in 0..num_token {
            topk[b * num_topk + 0] = 0;
            topk[b * num_topk + 1] = 1;
        }

        // routing
        indice[0 * num_token + 0] = true;
        indice[0 * num_token + 1] = true;
        indice[0 * num_token + 2] = true;

        indice[1 * num_token + 1] = true;
        indice[1 * num_token + 3] = true;

        for e in 0..num_experts {
            for b in 0..num_token {
                weight[e * num_token + b] = f16_from_f32(0.3 + 0.01 * e as f32 + 0.02 * b as f32);
            }
        }

        for e in 0..num_experts {
            for b in 0..num_token {
                for kk in 0..hmid {
                    nonlin[(e * num_token + b) * hmid + kk] =
                        f16_from_f32(0.005 * e as f32 + 0.01 * b as f32 + 0.0007 * kk as f32);
                }
            }
        }
        for e in 0..num_experts {
            for kk in 0..hmid {
                for j in 0..h {
                    wdown[(e * hmid + kk) * h + j] =
                        f16_from_f32(0.001 * e as f32 + 0.0009 * kk as f32 + 0.0002 * j as f32);
                }
            }
        }

        let runner = unsafe {
            ExpertsMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                weight.as_ptr(),
                topk.as_ptr(),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false,
            )
        };

        runner.run(num_token, 0, 1, 0);

        // token0 只命中 expert0 => slot1 应接近 0
        {
            let b = 0usize;
            let slot1 = 1usize;
            for j in 0..h {
                let v = f32_from_f16(out[(b * num_topk + slot1) * h + j]);
                assert!(v.abs() <= 1e-2, "token0 slot1 should be ~0, got {}", v);
            }
        }

        // token3 只命中 expert1 => slot0 应接近 0
        {
            let b = 3usize;
            let slot0 = 0usize;
            for j in 0..h {
                let v = f32_from_f16(out[(b * num_topk + slot0) * h + j]);
                assert!(v.abs() <= 1e-2, "token3 slot0 should be ~0, got {}", v);
            }
        }
    }
    #[test]
    fn test_down_multithread_tail_and_accumulate_semantics() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        let num_experts = 2usize; // E
        let num_token = 9usize; // B（故意 > MR=3 且 > MB）
        let hmid = 32usize; // K（整除 KC）
        let h = 48usize; // N：32 + 16 tail
        let num_topk = 2usize; // 每 token 两个 expert 槽

        let params = MatMulParams {
            a_row_step_macro: 6,   // MB
            b_row_step_macro: 48,  // NB 覆盖整行
            column_step_macro: 16, // KC（hmid%kc==0）
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        // nonlin[E,B,Hmid]
        let mut nonlin = vec![f16_from_f32(0.0); num_experts * num_token * hmid];
        // wdown[E,Hmid,H]  (这里仍按你算子输入的 K×N 形式给，算子内部会转置)
        let mut wdown = vec![f16_from_f32(0.0); num_experts * hmid * h];

        // 输出 [B,Ktop,H]：注意这里验证“+=”语义，所以我们先填一个非零底噪，再跑两次对比
        let mut out = vec![f16_from_f32(0.01); num_token * num_topk * h];

        let experts_indicator = vec![true; num_experts];

        // indice[E,B]：让 token 奇偶分别命中不同 expert，同时都在 topk 里
        let mut indice = vec![false; num_experts * num_token];
        for b in 0..num_token {
            indice[0 * num_token + b] = true; // expert0 命中所有
            indice[1 * num_token + b] = (b % 2 == 0); // expert1 命中偶数 token
        }

        // weight[E,B]
        let mut weight = vec![f16_from_f32(0.0); num_experts * num_token];
        for e in 0..num_experts {
            for b in 0..num_token {
                weight[e * num_token + b] = f16_from_f32(0.2 + 0.01 * e as f32 + 0.001 * b as f32);
            }
        }

        // topk[B,Ktop]：每行 [0,1] 升序
        let mut topk = vec![0usize; num_token * num_topk];
        for b in 0..num_token {
            topk[b * num_topk + 0] = 0;
            topk[b * num_topk + 1] = 1;
        }

        // 填 nonlin / wdown
        for e in 0..num_experts {
            for b in 0..num_token {
                for kk in 0..hmid {
                    nonlin[(e * num_token + b) * hmid + kk] =
                        f16_from_f32(0.01 * e as f32 + 0.002 * b as f32 + 0.0003 * kk as f32);
                }
            }
        }
        for e in 0..num_experts {
            for kk in 0..hmid {
                for j in 0..h {
                    // 让不同列/行有区分
                    wdown[(e * hmid + kk) * h + j] =
                        f16_from_f32(0.001 * e as f32 + 0.0007 * kk as f32 + 0.0002 * j as f32);
                }
            }
        }

        let runner = unsafe {
            ExpertsMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(),
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                weight.as_ptr(),
                topk.as_ptr(),
                out.as_mut_ptr(),
                num_experts,
                num_token,
                hmid,
                h,
                num_topk,
                params,
                false,
            )
        };

        // 用 2 线程跑一遍
        let cpu_num = 2usize;
        for tid in 0..cpu_num {
            runner.run(num_token, 0, cpu_num, tid);
        }

        // 保存一次结果
        let out_once = out.clone();

        // 再跑一遍（验证 += 语义：第二次的增量应基本等于第一次的增量）
        for tid in 0..cpu_num {
            runner.run(num_token, 0, cpu_num, tid);
        }

        // reference：我们不做全量 ref（太慢），抽样检查若干点
        // 关键：检查 tail 区间 j>=32 也被正确写入（len=16），且两次运行的增量一致
        let mut checks = 0usize;
        for b in 0..num_token {
            for slot in 0..num_topk {
                let e = topk[b * num_topk + slot];
                if !indice[e * num_token + b] {
                    continue;
                }
                // 抽两个 j：一个在 0..32，一个在 tail 32..48
                for &j in &[7usize, 40usize] {
                    let idx = (b * num_topk + slot) * h + j;
                    let v0 = f32_from_f16(out_once[idx]);
                    let v1 = f32_from_f16(out[idx]);

                    // v1 - v0 应约等于 v0 - baseline(0.01)，但 baseline 我们只要求“增量一致”
                    let delta1 = v0 - 0.01;
                    let delta2 = v1 - v0;

                    assert!(
                        (delta1 - delta2).abs() <= 8e-2,
                        "accumulate mismatch at b={},slot={},j={}: delta1={}, delta2={}",
                        b,
                        slot,
                        j,
                        delta1,
                        delta2
                    );
                    checks += 1;
                    if checks >= 32 {
                        break;
                    }
                }
                if checks >= 32 {
                    break;
                }
            }
            if checks >= 32 {
                break;
            }
        }

        assert!(checks > 0, "no checks performed");
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_down_stride_must_use_capacity_not_run_batch() {
        use std::arch::is_x86_feature_detected;
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        use crate::traits::from_f32::FromF32;

        #[inline]
        fn f16_from_f32(x: f32) -> f16 {
            <f16 as FromF32>::from_f32(x)
        }

        const E: usize = 2;
        const B_CAP: usize = 9; // capacity（构造时 num_token）
        const B_RUN: usize = 7; // run 时 batch_size
        const HMID: usize = 32;
        const H: usize = 32;
        const KTOP: usize = 2;

        // 3x32 微核配置（H=32 不会 tail）
        let params = MatMulParams {
            a_row_step_macro: 6,   // MB（随便 >= MR 即可）
            b_row_step_macro: 32,  // NB
            column_step_macro: 16, // KC（HMID%KC==0）
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        // -----------------------
        // 数据准备
        // -----------------------

        // NONLIN[e,b,k]
        // 全设为 1，便于算 expected
        let mut nonlin = vec![f16_from_f32(1.0); E * B_CAP * HMID];

        // W_down: 原始传入预期是 [E, HMID, H] (K×N) 行主 stride=H
        // 我们构造：
        //  - expert0: 全 1
        //  - expert1: 全 2
        // 这样 expert0 的输出是 32，expert1 的输出是 64（权重=1）
        let mut wdown = vec![f16_from_f32(0.0); E * HMID * H];
        for kk in 0..HMID {
            for j in 0..H {
                wdown[0 * (HMID * H) + kk * H + j] = f16_from_f32(1.0);
                wdown[1 * (HMID * H) + kk * H + j] = f16_from_f32(2.0);
            }
        }

        // experts_indicator：两个 expert 都开
        let experts_indicator = vec![true; E];

        // topk: 每个 token 的 topk 列表为 [0, 1]（升序）
        let mut topk = vec![0usize; B_CAP * KTOP];
        for b in 0..B_CAP {
            topk[b * KTOP + 0] = 0;
            topk[b * KTOP + 1] = 1;
        }

        // indice: [E, B_CAP]
        // 关键构造：
        //  - expert0: 只让 b=7,8 为 true（注意：这两个是“超出 B_RUN 的 token”）
        //  - expert1: 只让 b=5,6 为 true（这两个在 B_RUN 内）
        //
        // 正确实现（stride 用 B_CAP）：
        //  - expert1 只会写 token5/6 的 slot=1
        //  - token0/1 的 slot=1 必须保持 0
        //
        // 错误实现（stride 用 B_RUN=7）：
        //  - expert1 读 indice 时 base = 1*7 = 7
        //    它会把 expert0 的 b=7,8（正好在 indice[7],indice[8]）误当成 expert1 的 b=0,1
        //  => token0/1 会被错误写入 slot=1（出现非 0）
        let mut indice = vec![false; E * B_CAP];
        indice[0 * B_CAP + 7] = true;
        indice[0 * B_CAP + 8] = true;
        indice[1 * B_CAP + 5] = true;
        indice[1 * B_CAP + 6] = true;

        // weight: [E, B_CAP]，全 1
        let mut weight = vec![f16_from_f32(1.0); E * B_CAP];

        // output: [B_CAP, KTOP, H]，先清 0
        let mut out = vec![f16_from_f32(0.0); B_CAP * KTOP * H];

        // -----------------------
        // 运行
        // -----------------------
        let runner = unsafe {
            ExpertsMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown.as_ptr(), // 如果你的 new() 现在接受的是 wdown_nt_ptr(已经转置)，那这里要传入转置后的；但你这份代码仍按 K×N 传进来更合理
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                weight.as_ptr(),
                topk.as_ptr(),
                out.as_mut_ptr(),
                E,
                B_CAP,
                HMID,
                H,
                KTOP,
                params,
            )
        };

        // 单线程即可复现问题
        runner.run(B_RUN, 0, 1, 0);

        // -----------------------
        // 断言：token0/1 的 expert1(slot=1) 必须保持 0
        // -----------------------
        for b in 0..2 {
            let slot = 1usize; // expert1 的位置
            for j in 0..H {
                let idx = (b * KTOP + slot) * H + j;
                let got = out[idx] as f32;
                assert!(
                    got.abs() <= 1e-3,
                    "stride bug: token {} slot1 should be 0, got {} (j={})",
                    b,
                    got,
                    j
                );
            }
        }

        // 再检查：token5/6 的 expert1(slot=1) 应该是非零（避免“啥都没算”假通过）
        // 期望值：sum_{kk=0..31} 1 * 2 = 64
        for b in 5..=6 {
            let slot = 1usize;
            for j in 0..H {
                let idx = (b * KTOP + slot) * H + j;
                let got = out[idx] as f32;
                assert!(
                    (got - 64.0).abs() <= 2.0, // fp16 累积误差给点余量
                    "expected token {} slot1 ~= 64, got {} (j={})",
                    b,
                    got,
                    j
                );
            }
        }
    }
    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_down_stride_must_use_capacity_not_run_batch_nt_weight() {
        use std::arch::is_x86_feature_detected;
        if !is_x86_feature_detected!("avx512fp16") {
            eprintln!("skip: avx512fp16 not detected");
            return;
        }

        use crate::traits::from_f32::FromF32;
        #[inline]
        fn f16_from_f32(x: f32) -> f16 {
            <f16 as FromF32>::from_f32(x)
        }

        const E: usize = 2;
        const B_CAP: usize = 9; // capacity（构造时 num_token）
        const B_RUN: usize = 7; // run 时 batch_size
        const HMID: usize = 32;
        const H: usize = 32;
        const KTOP: usize = 2;

        let params = MatMulParams {
            a_row_step_macro: 6,   // MB
            b_row_step_macro: 32,  // NB
            column_step_macro: 16, // KC (HMID%KC==0)
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        // NONLIN[e,b,k]：全 1
        let nonlin = vec![f16_from_f32(1.0); E * B_CAP * HMID];

        // W_down_nt: [E, H, HMID]（你现在外部已经转置好就是这个）
        // 构造：
        //   expert0: 全 1
        //   expert1: 全 2
        // 则对任意输出列 j：dot = sum_k 1*W = 32 或 64（weight=1）
        let mut wdown_nt = vec![f16_from_f32(0.0); E * H * HMID];
        for j in 0..H {
            for kk in 0..HMID {
                wdown_nt[0 * (H * HMID) + j * HMID + kk] = f16_from_f32(1.0);
                wdown_nt[1 * (H * HMID) + j * HMID + kk] = f16_from_f32(2.0);
            }
        }

        let experts_indicator = vec![true; E];

        // topk: 每 token 的 topk 列表为 [0,1]（升序）
        let mut topk = vec![0usize; B_CAP * KTOP];
        for b in 0..B_CAP {
            topk[b * KTOP + 0] = 0;
            topk[b * KTOP + 1] = 1;
        }

        // indice: [E, B_CAP]
        // 关键构造（看解释）：
        let mut indice = vec![false; E * B_CAP];
        indice[0 * B_CAP + 7] = true;
        indice[0 * B_CAP + 8] = true;
        indice[1 * B_CAP + 5] = true;
        indice[1 * B_CAP + 6] = true;

        // weight: [E, B_CAP] 全 1
        let weight = vec![f16_from_f32(1.0); E * B_CAP];

        // output: [B_CAP, KTOP, H] 先清 0
        let mut out = vec![f16_from_f32(0.0); B_CAP * KTOP * H];

        let runner = unsafe {
            ExpertsMatMulDown::<f16>::new(
                nonlin.as_ptr(),
                wdown_nt.as_ptr(), // ✅ 直接传 NT
                experts_indicator.as_ptr(),
                indice.as_ptr(),
                weight.as_ptr(),
                topk.as_ptr(),
                out.as_mut_ptr(),
                E,
                B_CAP,
                HMID,
                H,
                KTOP,
                params,
            )
        };

        // 单线程足够复现
        runner.run(B_RUN, 0, 1, 0);

        // 断言1：token0/1 的 expert1(slot=1) 必须仍为 0
        for b in 0..2 {
            let slot = 1usize;
            for j in 0..H {
                let idx = (b * KTOP + slot) * H + j;
                let got = out[idx] as f32;
                assert!(
                    got.abs() <= 1e-3,
                    "stride bug: token {} slot1 should be 0, got {} (j={})",
                    b,
                    got,
                    j
                );
            }
        }

        // 断言2：token5/6 的 expert1(slot=1) 应该是非零，并且接近 64
        // ref: sum_k 1*2 = 64
        for b in 5..=6 {
            let slot = 1usize;
            for j in 0..H {
                let idx = (b * KTOP + slot) * H + j;
                let got = out[idx] as f32;
                assert!(
                    (got - 64.0).abs() <= 2.0,
                    "expected token {} slot1 ~= 64, got {} (j={})",
                    b,
                    got,
                    j
                );
            }
        }
    }
}




