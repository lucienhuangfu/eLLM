// === runner/matmul_kqv.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::assign::assign;
use super::mul_trait::MatMulkqvTrait;

#[inline]
unsafe fn transpose_kxn_to_nxk<T: Copy + Default>(w_kxn: *const T, k: usize, n: usize) -> Box<[T]> {
    // 输入: W[K×N] 行主；输出: W_nt[N×K] 行主（行距=K）
    let mut out = vec![T::default(); n * k];
    for kk in 0..k {
        let src_row = w_kxn.add(kk * n);
        for jj in 0..n {
            *out.as_mut_ptr().add(jj * k + kk) = *src_row.add(jj);
        }
    }
    out.into_boxed_slice()
}

/// K/Q/V 三个 GEMM（不含 sequence 维度）
///
/// 约定:
/// - A:  [M×K]
/// - Wq: [K×Nq] （构造期会转成 [Nq×K]）
/// - Wk: [K×Nkv]（构造期转 [Nkv×K]）
/// - Wv: 同上
/// - Cq: [M×Nq]
/// - Ck: [M×Nkv]
/// - Cv: [M×Nkv]
///
/// 分块空间:
///   对 Q:  tiles_m_q = ceil(M/MB), tiles_n_q = ceil(Nq/NB)
///   对 KV: tiles_m_kv = ceil(M/MB), tiles_n_kv = ceil(Nkv/NB)
///
/// 三条路径各自用自己的 tiles 做 assign(total_tiles, cpu_num, thread_id)。
#[derive(Clone)]
pub struct MatMul3<T> {
    // A / W / C
    hidden_ptr: ConstPtr<T>,   // A[M×K]
    q_weight_ptr: ConstPtr<T>, // Wq_nt[Nq×K]
    q_state_ptr: MutPtr<T>,    // Cq[M×Nq]
    k_weight_ptr: ConstPtr<T>, // Wk_nt[Nkv×K]
    k_state_ptr: MutPtr<T>,    // Ck[M×Nkv]
    v_weight_ptr: ConstPtr<T>, // Wv_nt[Nkv×K]
    v_state_ptr: MutPtr<T>,    // Cv[M×Nkv]

    // RoPE 相位表（按 N 方向拉平或按 head×dim 展开）
    rope_ptr: ConstPtr<T>, // 由外部保证布局一致

    // 形状
    head_dim: usize, // 比如 128（要求 head_dim % 32 == 0）
    m_row: usize,    // M
    col: usize,      // K
    b_q_row: usize,  // Nq
    b_kv_row: usize, // Nkv

    // 分块参数（MB/NB/KC/MR/NR=32）
    pub params: MatMulParams,
    _marker: PhantomData<T>,

    // 持有转置后权重缓冲
    wq_buf: Box<[T]>,
    wk_buf: Box<[T]>,
    wv_buf: Box<[T]>,

    // 线程私有 KC×NR 面板池
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize,
}

impl<T> MatMul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    #[inline]
    pub unsafe fn new(
        hidden_ptr: *const T,
        q_weight_ptr: *const T,
        q_state_ptr: *mut T,
        k_weight_ptr: *const T,
        k_state_ptr: *mut T,
        v_weight_ptr: *const T,
        v_state_ptr: *mut T,
        rope_ptr: *const T,
        head_dim: usize,
        m_row: usize,
        col: usize,
        b_q_row: usize,
        b_kv_row: usize,
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize,
        b_row_step_micro: usize,
    ) -> Self {
        // 预转置 W: [K×N] -> [N×K]
        let wq_buf = transpose_kxn_to_nxk::<T>(q_weight_ptr, col, b_q_row);
        let wk_buf = transpose_kxn_to_nxk::<T>(k_weight_ptr, col, b_kv_row);
        let wv_buf = transpose_kxn_to_nxk::<T>(v_weight_ptr, col, b_kv_row);

        let q_weight_ptr = ConstPtr {
            ptr: wq_buf.as_ptr(),
        };
        let k_weight_ptr = ConstPtr {
            ptr: wk_buf.as_ptr(),
        };
        let v_weight_ptr = ConstPtr {
            ptr: wv_buf.as_ptr(),
        };

        // 线程面板池：threads × (KC×NR)；threads 只在 new() 用一次，不进 struct
        let kc = column_step_macro.max(1);
        let nr = b_row_step_micro.max(1);
        let b_panel_stride_elems = kc * nr;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let pool_len = threads * b_panel_stride_elems;
        let b_panel_pool = vec![T::default(); pool_len].into_boxed_slice();

        Self {
            hidden_ptr: ConstPtr { ptr: hidden_ptr },
            q_weight_ptr,
            q_state_ptr: MutPtr { ptr: q_state_ptr },
            k_weight_ptr,
            k_state_ptr: MutPtr { ptr: k_state_ptr },
            v_weight_ptr,
            v_state_ptr: MutPtr { ptr: v_state_ptr },

            rope_ptr: ConstPtr { ptr: rope_ptr },

            head_dim,
            m_row,
            col,
            b_q_row,
            b_kv_row,

            params: MatMulParams {
                a_row_step_macro,
                b_row_step_macro,
                column_step_macro,
                a_row_step_micro,
                b_row_step_micro,
            },
            _marker: PhantomData,

            wq_buf,
            wk_buf,
            wv_buf,

            b_panel_pool,
            b_panel_stride_elems,
        }
    }

    #[inline(always)]
    fn thread_b_panel_ptr(&self, thread_id: usize) -> *mut T {
        unsafe {
            self.b_panel_pool
                .as_ptr()
                .add(thread_id * self.b_panel_stride_elems) as *mut T
        }
    }

    /// 从 B_nt[N×K] 打 kc×32 连续面板
    #[inline(always)]
    unsafe fn pack_b_panel_from_bnt(
        &self,
        b_nt: *const T, // N×K 行主
        ldb_row: usize, // = K
        n0: usize,      // N 起点
        k0: usize,      // K 起点
        kc: usize,
        nr: usize,   // = 32
        out: *mut T, // 输出: kc×nr 行主（每行 32 连续）
    ) {
        for p in 0..kc {
            let dst_row = out.add(p * nr);
            let col_k = k0 + p;
            for lane in 0..nr {
                let n = n0 + lane;
                let src = b_nt.add(n * ldb_row + col_k); // b_nt[n, col_k]
                *dst_row.add(lane) = *src;
            }
        }
    }

    /// 某一条路径的 M×N×K 分块 + assign(total_tiles) + 3×32 GEMM +（可选）3×128 finalize
    ///
    /// 要求 MatMulkqvTrait 已经扩展为：
    ///   fn compute1(&self, a, b_panel, c, lda, ldc, kc);
    ///   fn compute2(&self, c_head, rope_head, ldc);
    #[inline(always)]
    unsafe fn gemm_one_path_tiles(
        &self,
        a: *const T,    // A[M×K]
        c: *mut T,      // C[M×N]
        b_nt: *const T, // B_nt[N×K]
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
        rope_base: *const T, // 与 C 同行布局的 RoPE，相同 N 方向
        cpu_num: usize,
        thread_id: usize,
        finalize: bool, // 是否做 RMS+RoPE（3×128）
    ) where
        Self: MatMulkqvTrait<T>,
    {
        let mb = self.params.a_row_step_macro;
        let nb = self.params.b_row_step_macro;
        let kc_block = self.params.column_step_macro;
        let mr = self.params.a_row_step_micro; // 3
        let nr = self.params.b_row_step_micro; // 32

        debug_assert_eq!(mr, 3);
        debug_assert_eq!(nr, 32);
        debug_assert!(k % kc_block == 0);
        debug_assert!(n % nr == 0);
        debug_assert!(self.head_dim % nr == 0); // 比如 128 = 4×32

        let tiles_m = (m + mb - 1) / mb;
        let tiles_n = (n + nb - 1) / nb;
        let total_tiles = tiles_m * tiles_n;

        let b_panel_ptr = self.thread_b_panel_ptr(thread_id);
        let head_dim = self.head_dim;

        if let Some((tb, te)) = assign(total_tiles, cpu_num, thread_id) {
            for t in tb..te {
                let tm = t / tiles_n;
                let tn = t % tiles_n;

                let m0 = tm * mb;
                let n0 = tn * nb;

                let m_blk = (m - m0).min(mb);
                let n_blk = (n - n0).min(nb);

                debug_assert!(m_blk % mr == 0);
                debug_assert!(n_blk % nr == 0);

                let mut k0 = 0;
                while k0 < k {
                    let kc_cur = kc_block.min(k - k0);

                    let mut nt = 0;
                    while nt < n_blk {
                        // 打一个 kc×32 面板
                        self.pack_b_panel_from_bnt(
                            b_nt,
                            k, // ldb_row_nt = K
                            n0 + nt,
                            k0,
                            kc_cur,
                            nr,
                            b_panel_ptr,
                        );

                        let mut mi = 0;
                        while mi < m_blk {
                            let a_tile = a.add((m0 + mi) * k + k0); // 3×kc
                            let c_tile = c.add((m0 + mi) * ldc + (n0 + nt)); // 3×32

                            // ✅ GEMM 微核：3×32
                            self.compute1(
                                a_tile,
                                b_panel_ptr as *const T,
                                c_tile,
                                k, // lda = K
                                ldc,
                                kc_cur,
                            );

                            // ✅ 在 K 尾部 + 这一块是某个 head 的最后 32 列 => 做 3×128 finalize
                            if finalize && (k0 + kc_cur == k) {
                                let global_col = n0 + nt; // 当前这块 32 的起始列
                                let offset_in_head = global_col % head_dim;

                                // 只有当 offset+32 == head_dim 时，这块是该 head 的最后 32 列
                                if offset_in_head + nr == head_dim {
                                    let head_col0 = global_col - offset_in_head; // head 的起始列

                                    let c_head_ptr = c.add((m0 + mi) * ldc + head_col0);
                                    let rope_head_ptr = rope_base.add(head_col0);

                                    // ✅ 这里调用 trait 的 compute2，内部去用 3×128 内核
                                    self.compute2(
                                        c_head_ptr, // 3×128 的起点
                                        rope_head_ptr,
                                        ldc,
                                    );
                                }
                            }

                            mi += mr;
                        }

                        nt += nr;
                    }
                    k0 += kc_cur;
                }
            }
        }
    }

    /// 入口：不再有 S 维度，只针对当前 A[M×K] 做一次 K/Q/V。
    pub fn run(
        &self,
        position_index: usize,
        position_interval: usize,
        batch_size: usize, // M
        cpu_num: usize,
        thread_id: usize,
    ) where
        Self: MatMulkqvTrait<T>,
    {
        unsafe {
            let m = batch_size;
            let k = self.col;
            let n_q = self.b_q_row;
            let n_kv = self.b_kv_row;

            debug_assert_eq!(m, self.m_row);

            let a_base = self.hidden_ptr.ptr;
            let cq_base = self.q_state_ptr.ptr;
            let ck_base = self.k_state_ptr.ptr;
            let cv_base = self.v_state_ptr.ptr;

            let wq_nt = self.q_weight_ptr.ptr;
            let wk_nt = self.k_weight_ptr.ptr;
            let wv_nt = self.v_weight_ptr.ptr;

            let ldq = n_q;
            let ldk = n_kv;
            let ldv = n_kv;

            let rope_base = self.rope_ptr.ptr;

            // === 1) V 路径：通常不做 RMS+RoPE ===
            self.gemm_one_path_tiles(
                a_base, cv_base, wv_nt, m, n_kv, k, ldv, rope_base, cpu_num, thread_id,
                false, // V 不 finalize
            );

            // === 2) K 路径：做 RMS+RoPE（3×128） ===
            self.gemm_one_path_tiles(
                a_base, ck_base, wk_nt, m, n_kv, k, ldk, rope_base, cpu_num, thread_id,
                true, // K finalize
            );

            // === 3) Q 路径：看你模型需要，这里暂时 true ===
            self.gemm_one_path_tiles(
                a_base, cq_base, wq_nt, m, n_q, k, ldq, rope_base, cpu_num, thread_id,
                true, // 如果 Q 不需要 RMS+RoPE，可以改成 false
            );
        }
    }
}
impl<T> MatMulkqvTrait<T> for MatMul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    #[inline]
    default fn compute1(
        &self,
        _a: *const T,
        _b_panel: *const T,
        _c: *mut T,
        _lda: usize,
        _ldc: usize,
        _kc: usize,
    ) {
        // generic 占位，不做事
    }

    #[inline]
    default fn compute2(&self, _c_head: *mut T, _rope_head: *const T, _ldc: usize) {
        // generic 占位，不做事
    }
}

// ===== f16 特化：真正调用 AVX-512 微核 =====
impl MatMulkqvTrait<f16> for MatMul3<f16> {
    #[inline]
    fn compute1(
        &self,
        a: *const f16,
        b_panel: *const f16,
        c: *mut f16,
        lda: usize,
        ldc: usize,
        kc: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            // 3×32 GEMM 微核
            crate::kernel::x86_64::f16_512::matmul_rms_complex::matmul_update_inplace_3x32_accum(
                a, b_panel, c, lda, ldc, kc,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            // TODO: fallback 标量/通用实现
        }
    }

    #[inline]
    fn compute2(
        &self,
        c_head: *mut f16,      // 指向 3×128 的起点（某个 head 的整行）
        rope_head: *const f16, // 对应这个 head 的 128 维 RoPE 相位
        ldc: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            let eps: f16 = 1e-6f32 as f16;
            crate::kernel::x86_64::f16_512::matmul_rms_complex::matmul_finalize_rmsnorm_rope_inplace_3x128(
                c_head,
                rope_head,
                ldc,
                eps,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            // TODO: fallback 实现
        }
    }
}

// ===== f32 占位版本（以后要的话再补微核） =====
impl MatMulkqvTrait<f32> for MatMul3<f32> {
    #[inline]
    fn compute1(
        &self,
        _a: *const f32,
        _b_panel: *const f32,
        _c: *mut f32,
        _lda: usize,
        _ldc: usize,
        _kc: usize,
    ) {
        // TODO: f32 版本
    }

    #[inline]
    fn compute2(&self, _c_head: *mut f32, _rope_head: *const f32, _ldc: usize) {
        // TODO: f32 版本
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn avail_threads_cap(cap: usize) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(cap)
            .max(1)
    }

    /// float32 reference: C[M×N] = A[M×K] * B[K×N]
    fn gemm_ref_f32(a: &[f16], b: &[f16], c: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (a[i * k + kk] as f32) * (b[kk * n + j] as f32);
                }
                c[i * n + j] = sum;
            }
        }
    }

    /// 统一跑一次 runner（顺序模拟多线程），避免真的开线程引入不确定性
    fn run_runner(
        runner: &MatMul3<f16>,
        m: usize,
        thread_num: usize,
    ) {
        for tid in 0..thread_num {
            runner.run(0, 1, m, thread_num, tid);
        }
    }

    /// 让 finalize 不触发：
    /// - finalize=true 也行，但只要 N < head_dim=128，并且不会覆盖到 head 的最后 32 片段，就不会进 compute2。
    /// 这里我们简单设置 Nq/Nkv 都 < 128，且 tile 不会触到 offset_in_head+32==128 的条件。
    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_small_single_tile() {
        const M: usize = 3;
        const K: usize = 64;
        const NQ: usize = 32;
        const NKV: usize = 64;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(8);

        let mut a = vec![0.0f16; M * K];
        let mut wq = vec![0.0f16; K * NQ];
        let mut wk = vec![0.0f16; K * NKV];
        let mut wv = vec![0.0f16; K * NKV];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let rope = vec![0.0f16; HEAD_DIM]; // 不触发 finalize 时内容无所谓

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (0.01f32 * (i as f32) + 0.001f32 * (kk as f32)) as f16;
            }
        }
        for kk in 0..K {
            for j in 0..NQ {
                wq[kk * NQ + j] = (0.02f32 * (kk as f32) + 0.003f32 * (j as f32)) as f16;
            }
            for j in 0..NKV {
                wk[kk * NKV + j] = (0.015f32 * (kk as f32) + 0.002f32 * (j as f32)) as f16;
                wv[kk * NKV + j] = (0.017f32 * (kk as f32) + 0.0025f32 * (j as f32)) as f16;
            }
        }

        let runner = unsafe {
            MatMul3::<f16>::new(
                a.as_ptr(),
                wq.as_ptr(),
                cq.as_mut_ptr(),
                wk.as_ptr(),
                ck.as_mut_ptr(),
                wv.as_ptr(),
                cv.as_mut_ptr(),
                rope.as_ptr(),
                HEAD_DIM,
                M,
                K,
                NQ,
                NKV,
                3,   // MB
                32,  // NB
                64,  // KC
                3,   // MR
                32,  // NR
            )
        };

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f32(&a, &wq, &mut cq_ref, M, K, NQ);
        gemm_ref_f32(&a, &wk, &mut ck_ref, M, K, NKV);
        gemm_ref_f32(&a, &wv, &mut cv_ref, M, K, NKV);

        for i in 0..M {
            for j in 0..NQ {
                assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 1e-1);
            }
            for j in 0..NKV {
                assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 1e-1);
                assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 1e-1);
            }
        }
    }

    /// multi-tile：M 多个 MB，N 多个 NB（更能覆盖 assign + tile 逻辑）
    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_multi_tile() {
        const M: usize = 12;   // 4 个 MB(=3)
        const K: usize = 64;
        const NQ: usize = 96;  // 3 个 NR(=32)
        const NKV: usize = 96;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(16);

        let mut a = vec![0.0f16; M * K];
        let mut wq = vec![0.0f16; K * NQ];
        let mut wk = vec![0.0f16; K * NKV];
        let mut wv = vec![0.0f16; K * NKV];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let rope = vec![0.0f16; HEAD_DIM];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
            }
        }
        for kk in 0..K {
            for j in 0..NQ {
                wq[kk * NQ + j] = (((kk * 5 + j * 11) % 23) as f32 * 0.01) as f16;
            }
            for j in 0..NKV {
                wk[kk * NKV + j] = (((kk * 3 + j * 7) % 29) as f32 * 0.01) as f16;
                wv[kk * NKV + j] = (((kk * 9 + j * 4) % 31) as f32 * 0.01) as f16;
            }
        }

        let runner = unsafe {
            MatMul3::<f16>::new(
                a.as_ptr(),
                wq.as_ptr(),
                cq.as_mut_ptr(),
                wk.as_ptr(),
                ck.as_mut_ptr(),
                wv.as_ptr(),
                cv.as_mut_ptr(),
                rope.as_ptr(),
                HEAD_DIM,
                M,
                K,
                NQ,
                NKV,
                6,   // MB: 让 tile 变化
                64,  // NB
                64,  // KC
                3,   // MR
                32,  // NR
            )
        };

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f32(&a, &wq, &mut cq_ref, M, K, NQ);
        gemm_ref_f32(&a, &wk, &mut ck_ref, M, K, NKV);
        gemm_ref_f32(&a, &wv, &mut cv_ref, M, K, NKV);

        for i in 0..M {
            for j in 0..NQ {
                assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 5e-1);
            }
            for j in 0..NKV {
                assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 5e-1);
                assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 5e-1);
            }
        }
    }

    /// KC split：K=128, KC=64 => 两个 k-block，覆盖累加路径
    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_kc_split() {
        const M: usize = 6;
        const K: usize = 128;
        const NQ: usize = 64;
        const NKV: usize = 64;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(8);

        let mut a = vec![0.0f16; M * K];
        let mut wq = vec![0.0f16; K * NQ];
        let mut wk = vec![0.0f16; K * NKV];
        let mut wv = vec![0.0f16; K * NKV];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let rope = vec![0.0f16; HEAD_DIM];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i + kk) % 17) as f32 * 0.01) as f16;
            }
        }
        for kk in 0..K {
            for j in 0..NQ {
                wq[kk * NQ + j] = (((kk * 2 + j) % 13) as f32 * 0.01) as f16;
            }
            for j in 0..NKV {
                wk[kk * NKV + j] = (((kk * 3 + j * 2) % 19) as f32 * 0.01) as f16;
                wv[kk * NKV + j] = (((kk * 5 + j * 3) % 23) as f32 * 0.01) as f16;
            }
        }

        let runner = unsafe {
            MatMul3::<f16>::new(
                a.as_ptr(),
                wq.as_ptr(),
                cq.as_mut_ptr(),
                wk.as_ptr(),
                ck.as_mut_ptr(),
                wv.as_ptr(),
                cv.as_mut_ptr(),
                rope.as_ptr(),
                HEAD_DIM,
                M,
                K,
                NQ,
                NKV,
                6,   // MB
                32,  // NB
                64,  // KC（分两段）
                3,   // MR
                32,  // NR
            )
        };

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f32(&a, &wq, &mut cq_ref, M, K, NQ);
        gemm_ref_f32(&a, &wk, &mut ck_ref, M, K, NKV);
        gemm_ref_f32(&a, &wv, &mut cv_ref, M, K, NKV);

        for i in 0..M {
            for j in 0..NQ {
                assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 5e-1);
            }
            for j in 0..NKV {
                assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 5e-1);
                assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 5e-1);
            }
        }
    }

    /// 稍大一点的尺寸（但仍可做 unit test）：覆盖更多 tile/累加
    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_medium() {
        const M: usize = 48;
        const K: usize = 256;
        const NQ: usize = 256;
        const NKV: usize = 256;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(16);

        let mut a = vec![0.0f16; M * K];
        
        let mut wq = vec![0.0f16; K * NQ];
        let mut wk = vec![0.0f16; K * NKV];
        let mut wv = vec![0.0f16; K * NKV];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let rope = vec![0.0f16; HEAD_DIM];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i * 3 + kk * 5) % 97) as f32 * 0.001) as f16;
            }
        }
        for kk in 0..K {
            for j in 0..NQ {
                wq[kk * NQ + j] = (((kk * 7 + j * 11) % 101) as f32 * 0.001) as f16;
            }
            for j in 0..NKV {
                wk[kk * NKV + j] = (((kk * 13 + j * 17) % 103) as f32 * 0.001) as f16;
                wv[kk * NKV + j] = (((kk * 19 + j * 23) % 107) as f32 * 0.001) as f16;
            }
        }

        let runner = unsafe {
            MatMul3::<f16>::new(
                a.as_ptr(),
                wq.as_ptr(),
                cq.as_mut_ptr(),
                wk.as_ptr(),
                ck.as_mut_ptr(),
                wv.as_ptr(),
                cv.as_mut_ptr(),
                rope.as_ptr(),
                HEAD_DIM,
                M,
                K,
                NQ,
                NKV,
                24,   // MB
                128,  // NB
                64,   // KC
                3,    // MR
                32,   // NR
            )
        };

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f32(&a, &wq, &mut cq_ref, M, K, NQ);
        gemm_ref_f32(&a, &wk, &mut ck_ref, M, K, NKV);
        gemm_ref_f32(&a, &wv, &mut cv_ref, M, K, NKV);

        // 更大规模，容差放宽
        for i in 0..M {
            for j in 0..NQ {
                assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 1.0);
            }
            for j in 0..NKV {
                assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 1.0);
                assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 1.0);
            }
        }
    }
}