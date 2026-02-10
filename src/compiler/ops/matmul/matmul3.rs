// === compiler/mul/matmul3.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use crate::compiler::assign::assign;
use crate::compiler::ops::traits::mul_trait::MatMulkqvTrait;

// 添加 generic kernel 的引用
use crate::kernel::scalar::complex_mul::complex_mul;
use crate::kernel::scalar::rms_norm::rms_norm;

/// K/Q/V 三个 GEMM（不含 sequence 维度）
///
/// 约定（改动后）:
/// - A:      [M×K]
/// - Wq_nt:  [Nq×K]（✅ 由外部保证已经是 N×K 行主）
/// - Wk_nt:  [Nkv×K]
/// - Wv_nt:  [Nkv×K]
/// - Cq:     [M×Nq]
/// - Ck:     [M×Nkv]
/// - Cv:     [M×Nkv]
///
/// 注意：本文件不再在 new() 中转置权重，也不再持有 wq_buf/wk_buf/wv_buf。
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

    // 线程私有 KC×NR 面板池
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize,
}

impl<T> MatMul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    #[inline]
    pub fn new(
        hidden_ptr: *const T,
        q_weight_ptr_nt: *const T, // ✅ 现在约定为 Wq_nt[Nq×K]
        q_state_ptr: *mut T,
        k_weight_ptr_nt: *const T, // ✅ Wk_nt[Nkv×K]
        k_state_ptr: *mut T,
        v_weight_ptr_nt: *const T, // ✅ Wv_nt[Nkv×K]
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
            q_weight_ptr: ConstPtr { ptr: q_weight_ptr_nt }, // ✅ 直接引用外部 N×K
            q_state_ptr: MutPtr { ptr: q_state_ptr },
            k_weight_ptr: ConstPtr { ptr: k_weight_ptr_nt }, // ✅
            k_state_ptr: MutPtr { ptr: k_state_ptr },
            v_weight_ptr: ConstPtr { ptr: v_weight_ptr_nt }, // ✅
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
        debug_assert!(self.head_dim % nr == 0);

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

                            self.compute1(
                                a_tile,
                                b_panel_ptr as *const T,
                                c_tile,
                                k, // lda = K
                                ldc,
                                kc_cur,
                            );

                            if finalize && (k0 + kc_cur == k) {
                                let global_col = n0 + nt;
                                let offset_in_head = global_col % head_dim;

                                if offset_in_head + nr == head_dim {
                                    let head_col0 = global_col - offset_in_head;

                                    let c_head_ptr = c.add((m0 + mi) * ldc + head_col0);
                                    let rope_head_ptr = rope_base.add(head_col0);

                                    self.compute2(c_head_ptr, rope_head_ptr, ldc);
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
    pub fn run(&self, prefill_size: usize, _decode_size: usize, thread_num: usize, thread_id: usize) where
    Self: MatMulkqvTrait<T>,
{
    unsafe {
        let m_run = prefill_size;

        let k = self.col;
        let n_q = self.b_q_row;
        let n_kv = self.b_kv_row;

        let mr = self.params.a_row_step_micro.max(1); // 期望=3
        debug_assert_eq!(mr, 3);

        // === 关键：向上 pad 到 3 的倍数，用于固定 3×32 微核 ===
        let m_pad = ((m_run + mr - 1) / mr) * mr;

        // self.m_row 在你们“new 预留更大”语义下，应当视为 capacity（m_max）
        debug_assert!(m_pad <= self.m_row);

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
            a_base, cv_base, wv_nt, m_pad, n_kv, k, ldv, rope_base, thread_num, thread_id, false,
        );

        // === 2) K 路径：做 RMS+RoPE（3×128） ===
        self.gemm_one_path_tiles(
            a_base, ck_base, wk_nt, m_pad, n_kv, k, ldk, rope_base, thread_num, thread_id, true,
        );

        // === 3) Q 路径 ===
        self.gemm_one_path_tiles(
            a_base, cq_base, wq_nt, m_pad, n_q, k, ldq, rope_base, thread_num, thread_id, true,
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
        // generic 占位
    }

    #[inline]
    default fn compute2(&self, _c_head: *mut T, _rope_head: *const T, _ldc: usize) {
        // generic 占位
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
            crate::kernel::x86_64::f16_512::matmul_rms_complex::matmul_update_inplace_3x32_accum(
                a, b_panel, c, lda, ldc, kc,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            // TODO: fallback
        }
    }

    #[inline]
    fn compute2(&self, c_head: *mut f16, rope_head: *const f16, ldc: usize) {
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
            // TODO: fallback
        }
    }
}

// ===== f32 占位版本（以后要的话再补微核） =====
impl MatMulkqvTrait<f32> for MatMul3<f32> {
    #[inline]
    fn compute1(
        &self,
        a: *const f32,
        b_panel: *const f32,
        c: *mut f32,
        lda: usize,
        ldc: usize,
        kc: usize,
    ) {
        unsafe {
            for m in 0..3 {
                for n in 0..32 {
                    let mut sum = 0.0;
                    for k in 0..kc {
                        let val_a = *a.add(m * lda + k);
                        let val_b = *b_panel.add(k * 32 + n);
                        sum += val_a * val_b;
                    }
                    *c.add(m * ldc + n) += sum;
                }
            }
        }
    }

    #[inline]
    fn compute2(&self, c_head: *mut f32, rope_head: *const f32, ldc: usize) {
        unsafe {
            let eps = 1e-6;
            for r in 0..3 {
                let row_ptr = c_head.add(r * ldc);
                rms_norm(row_ptr, row_ptr, 128, eps);
                complex_mul(row_ptr, rope_head, row_ptr, 128);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ========================================================================
    // Helpers for f32 tests
    // ========================================================================

    /// 参考：A[M×K] * W[K×N]，但我们存的是 W_nt[N×K]，所以用 w_nt[j*K + p]
    fn ref_matmul_f32_from_wnt(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        w_nt: &[f32], // N×K
        c: &mut [f32],
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * w_nt[j * k + p];
                }
                c[i * n + j] = sum;
            }
        }
    }

    fn ref_post_process_f32(m: usize, n: usize, c: &mut [f32], rope: &[f32], head_dim: usize) {
        let eps = 1e-6;
        unsafe {
            for i in 0..m {
                for h_base in (0..n).step_by(head_dim) {
                    let ptr = c.as_mut_ptr().add(i * n + h_base);
                    let rope_ptr = rope.as_ptr().add(h_base);
                    rms_norm(ptr, ptr, head_dim, eps);
                    complex_mul(ptr, rope_ptr, ptr, head_dim);
                }
            }
        }
    }

    // ========================================================================
    // Helpers for f16 tests
    // ========================================================================

    fn avail_threads_cap(cap: usize) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(cap)
            .max(1)
    }

    /// f32 accumulate reference GEMM: out = A[M×K] * W[K×N]，但 W 存的是 W_nt[N×K]
    fn gemm_ref_f16_acc_f32_from_wnt(
        a: &[f16],
        w_nt: &[f16], // N×K
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (a[i * k + kk] as f32) * (w_nt[j * k + kk] as f32);
                }
                out[i * n + j] = sum;
            }
        }
    }

    fn run_runner(runner: &MatMul3<f16>, m: usize, thread_num: usize) {
        for tid in 0..thread_num {
            runner.run(m, 0, thread_num, tid);
        }
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[test]
    fn test_matmul3_qkv_f32_72_rows() {
        let m = 72;
        let k = 256;
        let head_dim = 128;
        let n_q = 32 * 128; // 4096
        let n_kv = 4 * 128; // 512

        // A
        let mut a = vec![0.0f32; m * k];

        // ✅ 权重改为 N×K（W_nt）
        let mut wq_nt = vec![0.0f32; n_q * k];
        let mut wk_nt = vec![0.0f32; n_kv * k];
        let mut wv_nt = vec![0.0f32; n_kv * k];

        let mut cq = vec![0.0f32; m * n_q];
        let mut cq_ref = vec![0.0f32; m * n_q];

        let mut ck = vec![0.0f32; m * n_kv];
        let mut ck_ref = vec![0.0f32; m * n_kv];

        let mut cv = vec![0.0f32; m * n_kv];
        let mut cv_ref = vec![0.0f32; m * n_kv];

        let mut rope = vec![1.0f32; n_q.max(n_kv)];

        for i in 0..m * k {
            a[i] = (i % 100) as f32 * 0.01;
        }

        // 原先是 K×N 的填法，这里改成 N×K
        for j in 0..n_q {
            for kk in 0..k {
                let idx_old = kk * n_q + j;
                let v = ((idx_old + 1) % 7) as f32 * 0.01;
                wq_nt[j * k + kk] = v;
            }
        }
        for j in 0..n_kv {
            for kk in 0..k {
                let idx_old = kk * n_kv + j;
                wk_nt[j * k + kk] = ((idx_old + 2) % 7) as f32 * 0.01;
                wv_nt[j * k + kk] = ((idx_old + 3) % 7) as f32 * 0.01;
            }
        }
        for i in 0..rope.len() {
            rope[i] = 1.0;
        }

        unsafe {
            let matmul = MatMul3::<f32>::new(
                a.as_ptr(),
                wq_nt.as_ptr(),
                cq.as_mut_ptr(),
                wk_nt.as_ptr(),
                ck.as_mut_ptr(),
                wv_nt.as_ptr(),
                cv.as_mut_ptr(),
                rope.as_ptr(),
                head_dim,
                m,
                k,
                n_q,
                n_kv,
                24,  // MB
                128, // NB
                32,  // KC
                3,   // MR
                32,  // NR
            );

            matmul.run(m, 0, 1, 0);

            // reference（从 W_nt 计算）
            ref_matmul_f32_from_wnt(m, k, n_q, &a, &wq_nt, &mut cq_ref);
            ref_post_process_f32(m, n_q, &mut cq_ref, &rope, head_dim);

            ref_matmul_f32_from_wnt(m, k, n_kv, &a, &wk_nt, &mut ck_ref);
            ref_post_process_f32(m, n_kv, &mut ck_ref, &rope, head_dim);

            ref_matmul_f32_from_wnt(m, k, n_kv, &a, &wv_nt, &mut cv_ref);

            let verify = |name: &str, out: &[f32], reference: &[f32]| {
                let mut max_diff = 0.0f32;
                for (i, (v1, v2)) in out.iter().zip(reference.iter()).enumerate() {
                    let diff = (v1 - v2).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    if diff > 1e-3 {
                        panic!(
                            "{} mismatch at index {}: got {}, expected {}, diff {}",
                            name, i, v1, v2, diff
                        );
                    }
                }
                println!("{} passed. Max diff: {}", name, max_diff);
            };

            verify("Q Output", &cq, &cq_ref);
            verify("K Output", &ck, &ck_ref);
            verify("V Output", &cv, &cv_ref);
        }
    }

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

        // ✅ 权重改为 N×K
        let mut wq_nt = vec![0.0f16; NQ * K];
        let mut wk_nt = vec![0.0f16; NKV * K];
        let mut wv_nt = vec![0.0f16; NKV * K];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let rope = vec![0.0f16; HEAD_DIM];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (0.01f32 * (i as f32) + 0.001f32 * (kk as f32)) as f16;
            }
        }

        // 原来是 K×N 的写法，这里转成 N×K
        for j in 0..NQ {
            for kk in 0..K {
                wq_nt[j * K + kk] = (0.02f32 * (kk as f32) + 0.003f32 * (j as f32)) as f16;
            }
        }
        for j in 0..NKV {
            for kk in 0..K {
                wk_nt[j * K + kk] = (0.015f32 * (kk as f32) + 0.002f32 * (j as f32)) as f16;
                wv_nt[j * K + kk] = (0.017f32 * (kk as f32) + 0.0025f32 * (j as f32)) as f16;
            }
        }

        let runner = MatMul3::<f16>::new(
            a.as_ptr(),
            wq_nt.as_ptr(),
            cq.as_mut_ptr(),
            wk_nt.as_ptr(),
            ck.as_mut_ptr(),
            wv_nt.as_ptr(),
            cv.as_mut_ptr(),
            rope.as_ptr(),
            HEAD_DIM,
            M,
            K,
            NQ,
            NKV,
            3,  // MB
            32, // NB
            64, // KC
            3,  // MR
            32, // NR
        );

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M, K, NQ);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M, K, NKV);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M, K, NKV);

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

    // 下面这些 f16 测试原本都以 K×N 构造权重并 reference 也是 K×N；
    // 现在统一改为 N×K（W_nt），因此全部按同样方式改造：
    //   1) wq/wk/wv 的 Vec 长度从 K*N 变为 N*K
    //   2) 填充索引从 [kk*N + j] 变为 [j*K + kk]
    //   3) reference 从 b[kk*N + j] 变为 w_nt[j*K + kk]
    //
    // 为了避免篇幅爆炸，这里保留你原测试结构，并做同样的 layout 改动。

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_multi_tile() {
        const M: usize = 12;
        const K: usize = 64;
        const NQ: usize = 96;
        const NKV: usize = 96;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(16);

        let mut a = vec![0.0f16; M * K];
        let mut wq_nt = vec![0.0f16; NQ * K];
        let mut wk_nt = vec![0.0f16; NKV * K];
        let mut wv_nt = vec![0.0f16; NKV * K];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let rope = vec![0.0f16; HEAD_DIM];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i * 7 + kk * 3) % 19) as f32 * 0.01f32) as f16;
            }
        }
        for j in 0..NQ {
            for kk in 0..K {
                wq_nt[j * K + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.01f32) as f16;
            }
        }
        for j in 0..NKV {
            for kk in 0..K {
                wk_nt[j * K + kk] = (((kk * 3 + j * 7) % 29) as f32 * 0.01f32) as f16;
                wv_nt[j * K + kk] = (((kk * 9 + j * 4) % 31) as f32 * 0.01f32) as f16;
            }
        }

        let runner = MatMul3::<f16>::new(
            a.as_ptr(),
            wq_nt.as_ptr(),
            cq.as_mut_ptr(),
            wk_nt.as_ptr(),
            ck.as_mut_ptr(),
            wv_nt.as_ptr(),
            cv.as_mut_ptr(),
            rope.as_ptr(),
            HEAD_DIM,
            M,
            K,
            NQ,
            NKV,
            6,
            64,
            64,
            3,
            32,
        );

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M, K, NQ);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M, K, NKV);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M, K, NKV);

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

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_kqv_f16_avx512_kc_split() {
        const M: usize = 3;
        const K: usize = 128;
        const NQ: usize = 64;
        const NKV: usize = 64;
        const HEAD_DIM: usize = 128;

        let thread_num = avail_threads_cap(8);

        let mut a = vec![0.0f16; M * K];
        let mut wq_nt = vec![0.0f16; NQ * K];
        let mut wk_nt = vec![0.0f16; NKV * K];
        let mut wv_nt = vec![0.0f16; NKV * K];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let rope = vec![0.0f16; HEAD_DIM];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i + kk) % 17) as f32 * 0.01f32) as f16;
            }
        }
        for j in 0..NQ {
            for kk in 0..K {
                wq_nt[j * K + kk] = (((kk * 2 + j) % 13) as f32 * 0.01f32) as f16;
            }
        }
        for j in 0..NKV {
            for kk in 0..K {
                wk_nt[j * K + kk] = (((kk * 3 + j * 2) % 19) as f32 * 0.01f32) as f16;
                wv_nt[j * K + kk] = (((kk * 5 + j * 3) % 23) as f32 * 0.01f32) as f16;
            }
        }

        let runner = MatMul3::<f16>::new(
            a.as_ptr(),
            wq_nt.as_ptr(),
            cq.as_mut_ptr(),
            wk_nt.as_ptr(),
            ck.as_mut_ptr(),
            wv_nt.as_ptr(),
            cv.as_mut_ptr(),
            rope.as_ptr(),
            HEAD_DIM,
            M,
            K,
            NQ,
            NKV,
            3,
            32,
            64,
            3,
            32,
        );

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M, K, NQ);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M, K, NKV);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M, K, NKV);

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
        let mut wq_nt = vec![0.0f16; NQ * K];
        let mut wk_nt = vec![0.0f16; NKV * K];
        let mut wv_nt = vec![0.0f16; NKV * K];

        let mut cq = vec![0.0f16; M * NQ];
        let mut ck = vec![0.0f16; M * NKV];
        let mut cv = vec![0.0f16; M * NKV];

        let rope = vec![0.0f16; HEAD_DIM];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i * 3 + kk * 5) % 97) as f32 * 0.001) as f16;
            }
        }
        for j in 0..NQ {
            for kk in 0..K {
                wq_nt[j * K + kk] = (((kk * 7 + j * 11) % 101) as f32 * 0.001) as f16;
            }
        }
        for j in 0..NKV {
            for kk in 0..K {
                wk_nt[j * K + kk] = (((kk * 13 + j * 17) % 103) as f32 * 0.001) as f16;
                wv_nt[j * K + kk] = (((kk * 19 + j * 23) % 107) as f32 * 0.001) as f16;
            }
        }

        let runner = MatMul3::<f16>::new(
            a.as_ptr(),
            wq_nt.as_ptr(),
            cq.as_mut_ptr(),
            wk_nt.as_ptr(),
            ck.as_mut_ptr(),
            wv_nt.as_ptr(),
            cv.as_mut_ptr(),
            rope.as_ptr(),
            HEAD_DIM,
            M,
            K,
            NQ,
            NKV,
            24,
            128,
            64,
            3,
            32,
        );

        run_runner(&runner, M, thread_num);

        let mut cq_ref = vec![0.0f32; M * NQ];
        let mut ck_ref = vec![0.0f32; M * NKV];
        let mut cv_ref = vec![0.0f32; M * NKV];
        gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M, K, NQ);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M, K, NKV);
        gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M, K, NKV);

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

    #[test]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
fn test_kqv_f16_avx512_batch7_pad_to9_no_finalize() {
    use approx::assert_abs_diff_eq;

    const M_RUN: usize = 7;   // 非 3 倍数
    const M_MAX: usize = 9;   // ceil_div(7,3)*3 = 9
    const K: usize = 64;

    // 关键：让 N < HEAD_DIM，避免 finalize 条件 offset+32==head_dim 触发
    const HEAD_DIM: usize = 128;
    const NQ: usize = 64;   // < 128 => nt 只会 0,32，不会到 96
    const NKV: usize = 64;  // < 128

    let thread_num = avail_threads_cap(8);

    // A 按 M_MAX 分配（capacity），只填前 M_RUN 行，其余保持 0
    let mut a = vec![0.0f16; M_MAX * K];
    for i in 0..M_RUN {
        for kk in 0..K {
            a[i * K + kk] = (0.01f32 * (i as f32) + 0.001f32 * (kk as f32)) as f16;
        }
    }

    // W_nt: N×K
    let mut wq_nt = vec![0.0f16; NQ * K];
    let mut wk_nt = vec![0.0f16; NKV * K];
    let mut wv_nt = vec![0.0f16; NKV * K];

    for j in 0..NQ {
        for kk in 0..K {
            wq_nt[j * K + kk] = (0.02f32 * (kk as f32) + 0.003f32 * (j as f32)) as f16;
        }
    }
    for j in 0..NKV {
        for kk in 0..K {
            wk_nt[j * K + kk] = (0.015f32 * (kk as f32) + 0.002f32 * (j as f32)) as f16;
            wv_nt[j * K + kk] = (0.017f32 * (kk as f32) + 0.0025f32 * (j as f32)) as f16;
        }
    }

    // 输出按 M_MAX 分配（capacity）
    let mut cq = vec![0.0f16; M_MAX * NQ];
    let mut ck = vec![0.0f16; M_MAX * NKV];
    let mut cv = vec![0.0f16; M_MAX * NKV];

    // rope 给足长度即可（本测试不会触发 finalize）
    let rope = vec![1.0f16; HEAD_DIM.max(NQ).max(NKV)];

    // MB 要是 MR 的倍数（你现有 gemm_one_path_tiles 要求 m_blk%mr==0）
    let runner = MatMul3::<f16>::new(
        a.as_ptr(),
        wq_nt.as_ptr(),
        cq.as_mut_ptr(),
        wk_nt.as_ptr(),
        ck.as_mut_ptr(),
        wv_nt.as_ptr(),
        cv.as_mut_ptr(),
        rope.as_ptr(),
        HEAD_DIM,
        M_MAX, // m_row 当 capacity 用
        K,
        NQ,
        NKV,
        6,   // MB
        64,  // NB（=N）
        64,  // KC（=K）
        3,   // MR
        32,  // NR
    );

    // run 传 batch=7（内部 pad 到 9）
    run_runner(&runner, M_RUN, thread_num);

    // reference：只算前 7 行（pad 行不关心）
    let mut cq_ref = vec![0.0f32; M_RUN * NQ];
    let mut ck_ref = vec![0.0f32; M_RUN * NKV];
    let mut cv_ref = vec![0.0f32; M_RUN * NKV];

    gemm_ref_f16_acc_f32_from_wnt(&a, &wq_nt, &mut cq_ref, M_RUN, K, NQ);
    gemm_ref_f16_acc_f32_from_wnt(&a, &wk_nt, &mut ck_ref, M_RUN, K, NKV);
    gemm_ref_f16_acc_f32_from_wnt(&a, &wv_nt, &mut cv_ref, M_RUN, K, NKV);

    for i in 0..M_RUN {
        for j in 0..NQ {
            assert_abs_diff_eq!(cq[i * NQ + j] as f32, cq_ref[i * NQ + j], epsilon = 5e-1);
        }
        for j in 0..NKV {
            assert_abs_diff_eq!(ck[i * NKV + j] as f32, ck_ref[i * NKV + j], epsilon = 5e-1);
            assert_abs_diff_eq!(cv[i * NKV + j] as f32, cv_ref[i * NKV + j], epsilon = 5e-1);
        }
    }
}
}


