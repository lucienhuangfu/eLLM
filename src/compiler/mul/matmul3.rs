// === runner/matmul_kqv.rs ===
#![allow(non_snake_case)]

use std::cell::Cell;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    send_sync_ptr::{ConstPtr, MutPtr},
    matmul_params::MatmulParams,
};
use super::super::assign::assign;
use super::mul_trait::Matmul4Trait; // 外部提供：compute1/compute2 的 trait

/// Q/K/V 线性（不打包 B），Q/K tile 尾部做 RMS + RoPE（原位）
/// 约定：MR=3，NR=128，head_dim=128，N 是 128 的倍数
#[derive(Clone)]
pub struct Matmul3<T> {
    // A / W / C
    hidden_ptr: ConstPtr<T>,   // A[S×M×K]
    q_weight_ptr: ConstPtr<T>, // Wq[K×N]
    q_state_ptr: MutPtr<T>,    // Cq[S×M×N]
    k_weight_ptr: ConstPtr<T>, // Wk[K×N]
    k_state_ptr: MutPtr<T>,    // Ck[S×M×N]
    v_weight_ptr: ConstPtr<T>, // Wv[K×N]
    v_state_ptr: MutPtr<T>,    // Cv[S×M×N]

    // RoPE 相位表：[max_seq_len × head_dim]，交错 (cos0, sin0, cos1, sin1, ...)
    position_embedding_ptr: ConstPtr<T>,

    // 形状
    head_dim: usize, // =128（必须偶数）
    a_h_row: usize,  // M
    col: usize,      // K
    b_q_row: usize,  // N (Wq 同形)
    b_kv_row: usize, // N (Wk/Wv 同形)

    // 分块（仅承载 MB/NB/KC/MR/NR）
    pub params: MatmulParams,
    _marker: PhantomData<T>,

    // —— compute1/2 从“本地字段”取调用形状 —— //
    active_ldc: Cell<usize>, // 当前路径的 N（Q/K/V）
    lda_fixed: usize,        // = K
    kc_fixed: usize,         // = params.column_step_macro
    mr_fixed: usize,         // = params.a_row_step_micro (应=3)
    nr_fixed: usize,         // = params.b_row_step_micro (应=128)
}

impl<T> Matmul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    #[inline]
    pub fn new(
        hidden_ptr: *const T,
        q_weight_ptr: *const T,
        q_state_ptr: *mut T,
        k_weight_ptr: *const T,
        k_state_ptr: *mut T,
        v_weight_ptr: *const T,
        v_state_ptr: *mut T,
        position_embedding_ptr: *const T,
        head_dim: usize,
        a_h_row: usize,  // M
        col: usize,      // K
        b_q_row: usize,  // N
        b_kv_row: usize, // N
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize, // 3
        b_row_step_micro: usize, // 128
    ) -> Self {
        Self {
            hidden_ptr: ConstPtr { ptr: hidden_ptr },
            q_weight_ptr: ConstPtr { ptr: q_weight_ptr },
            q_state_ptr: MutPtr { ptr: q_state_ptr },
            k_weight_ptr: ConstPtr { ptr: k_weight_ptr },
            k_state_ptr: MutPtr { ptr: k_state_ptr },
            v_weight_ptr: ConstPtr { ptr: v_weight_ptr },
            v_state_ptr: MutPtr { ptr: v_state_ptr },
            position_embedding_ptr: ConstPtr {
                ptr: position_embedding_ptr,
            },
            head_dim,
            a_h_row,
            col,
            b_q_row,
            b_kv_row,
            params: MatmulParams {
                a_row_step_macro,
                b_row_step_macro,
                column_step_macro,
                a_row_step_micro,
                b_row_step_micro,
            },
            _marker: PhantomData,
            active_ldc: Cell::new(0),
            lda_fixed: col,
            kc_fixed: column_step_macro,
            mr_fixed: a_row_step_micro,
            nr_fixed: b_row_step_micro,
        }
    }

    /// 不打包 B：B 为行主 K×N。内核 compute1 直接从 B 的行上读 128 宽向量。
    /// Q/K 在该 tile 的最后一个 kc 后执行 compute2（RMSNorm weight=1 + RoPE），原位。
    /// V 不做 compute2。
    pub fn run(
        &self,
        position_index: usize,
        position_interval: usize,
        batch_size: usize, // M
        cpu_num: usize,
        thread_id: usize,
    ) {
        /*
        unsafe {
            // 维度
            let m = batch_size;
            let k = self.col;
            let n = self.b_q_row;

            // 分块
            let mb = self.params.a_row_step_macro.max(1);
            let nb = self.params.b_row_step_macro.max(1);
            let kc = self.params.column_step_macro.max(1);
            let mr = self.params.a_row_step_micro.max(1);
            let nr = self.params.b_row_step_micro.max(1);

            debug_assert_eq!(mr, 3, "MR must be 3");
            debug_assert_eq!(nr, 128, "NR must be 128");
            debug_assert!(k % kc == 0, "K must be divisible by KC");
            debug_assert!(n % nr == 0, "N must be multiple of 128");
            debug_assert_eq!(self.head_dim, 128, "head_dim must be 128");

            // 指针与行距
            let a_base = self.hidden_ptr.ptr; // A[S×M×K]，lda = k
            let lda = k;

            let wq = self.q_weight_ptr.ptr; // K×N 行主
            let wk = self.k_weight_ptr.ptr;
            let wv = self.v_weight_ptr.ptr;

            let cq_base = self.q_state_ptr.ptr;
            let ldq = n;
            let ck_base = self.k_state_ptr.ptr;
            let ldk = n;
            let cv_base = self.v_state_ptr.ptr;
            let ldv = n;

            // RoPE 预表（交错），行距=head_dim=128
            let rope_tab = self.position_embedding_ptr.ptr;
            let rope_stride = self.head_dim;

            // 序列范围
            let s_begin = position_index;
            let s_end = position_index + position_interval;
            let s_len = s_end - s_begin;

            // 把 S 维切片给线程
            if let Some((tb, te)) = assign(s_len, cpu_num, thread_id) {
                for s_rel in tb..te {
                    let s = s_begin + s_rel;
                    let a_s = a_base.add(s * m * k);
                    let rope_row_ptr = rope_tab.add(s * rope_stride);

                    // ===== Q 路径 =====
                    self.active_ldc.set(ldq);
                    {
                        let c_s = cq_base.add(s * m * n);
                        let tiles_m = (m + mb - 1) / mb;
                        let tiles_n = (n + nb - 1) / nb;

                        for tm in 0..tiles_m {
                            for tn in 0..tiles_n {
                                let m0 = tm * mb;
                                let n0 = tn * nb;

                                let m_blk = (m - m0).min(mb);
                                let n_blk = (n - n0).min(nb);
                                debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                                let mut k0 = 0;
                                while k0 < k {
                                    let mut nt = 0;
                                    while nt < n_blk {
                                        let mut mi = 0;
                                        while mi < m_blk {
                                            let a_tile = a_s.add((m0 + mi) * lda + k0); // 3×kc
                                            let c_tile = c_s.add((m0 + mi) * ldq + (n0 + nt)); // 3×128
                                            let b_row = wq.add(k0 * ldq + (n0 + nt)); // B[k0, n0+nt]

                                            // 3×128 累加
                                            self.compute1(a_tile, b_row, c_tile);

                                            // 尾 kc：RMS + RoPE（原位）
                                            if k0 + kc == k {
                                                self.compute2(c_tile, rope_row_ptr);
                                            }

                                            mi += mr;
                                        }
                                        nt += nr;
                                    }
                                    k0 += kc;
                                }
                            }
                        }
                    }

                    // ===== K 路径 =====
                    self.active_ldc.set(ldk);
                    {
                        let c_s = ck_base.add(s * m * n);
                        let tiles_m = (m + mb - 1) / mb;
                        let tiles_n = (n + nb - 1) / nb;

                        for tm in 0..tiles_m {
                            for tn in 0..tiles_n {
                                let m0 = tm * mb;
                                let n0 = tn * nb;

                                let m_blk = (m - m0).min(mb);
                                let n_blk = (n - n0).min(nb);
                                debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                                let mut k0 = 0;
                                while k0 < k {
                                    let mut nt = 0;
                                    while nt < n_blk {
                                        let mut mi = 0;
                                        while mi < m_blk {
                                            let a_tile = a_s.add((m0 + mi) * lda + k0);
                                            let c_tile = c_s.add((m0 + mi) * ldk + (n0 + nt));
                                            let b_row = wk.add(k0 * ldk + (n0 + nt));

                                            self.compute1(a_tile, b_row, c_tile);

                                            if k0 + kc == k {
                                                self.compute2(c_tile, rope_row_ptr);
                                            }

                                            mi += mr;
                                        }
                                        nt += nr;
                                    }
                                    k0 += kc;
                                }
                            }
                        }
                    }

                    // ===== V 路径（无 finalize）=====
                    self.active_ldc.set(ldv);
                    {
                        let c_s = cv_base.add(s * m * n);
                        let tiles_m = (m + mb - 1) / mb;
                        let tiles_n = (n + nb - 1) / nb;

                        for tm in 0..tiles_m {
                            for tn in 0..tiles_n {
                                let m0 = tm * mb;
                                let n0 = tn * nb;

                                let m_blk = (m - m0).min(mb);
                                let n_blk = (n - n0).min(nb);
                                debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                                let mut k0 = 0;
                                while k0 < k {
                                    let mut nt = 0;
                                    while nt < n_blk {
                                        let mut mi = 0;
                                        while mi < m_blk {
                                            let a_tile = a_s.add((m0 + mi) * lda + k0);
                                            let c_tile = c_s.add((m0 + mi) * ldv + (n0 + nt));
                                            let b_row = wv.add(k0 * ldv + (n0 + nt));

                                            self.compute1(a_tile, b_row, c_tile);
                                            // V 不做 finalize

                                            mi += mr;
                                        }
                                        nt += nr;
                                    }
                                    k0 += kc;
                                }
                            }
                        }
                    }
                }
            }
        } */
    }
}
impl<T> Matmul4Trait<T> for Matmul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute1(&self, _a: *const T, _b: *const T, _c: *mut T) {
        // generic 占位，暂时什么都不做
    }

    default fn compute2(&self, _c: *mut T, _rope_ptr: *const T) {
        // generic 占位
    }
}

impl Matmul4Trait<f16> for Matmul3<f16> {
    fn compute1(&self, a: *const f16, b_row: *const f16, c: *mut f16) {
        let call_param = MatmulParams {
            a_row_step_macro: self.lda_fixed,
            b_row_step_macro: self.active_ldc.get(),
            column_step_macro: self.kc_fixed,
            a_row_step_micro: self.mr_fixed, // 3
            b_row_step_micro: self.nr_fixed, // 128
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            crate::kernel::x86_64::f16_512::Matmul_rms_complex::Matmul_update_inplace_3x128_accum(
                a,
                b_row,
                c,
                &call_param,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            // fallback generic 或者留空
        }
    }

    fn compute2(&self, c: *mut f16, rope_ptr: *const f16) {
        let call_param = MatmulParams {
            a_row_step_macro: self.lda_fixed,
            b_row_step_macro: self.active_ldc.get(),
            column_step_macro: self.kc_fixed,
            a_row_step_micro: self.mr_fixed,
            b_row_step_micro: self.nr_fixed,
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            let eps: f16 = 1e-6f32 as f16;
            crate::kernel::x86_64::f16_512::matmul_rms_complex::matmul_finalize_rmsnorm_rope_inplace_3x128(
                c, rope_ptr, eps, &call_param,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            // fallback generic 或者留空
        }
    }
}

impl Matmul4Trait<f32> for Matmul3<f32> {
    fn compute1(&self, _a: *const f32, _b: *const f32, _c: *mut f32) {
        // TODO: f32 版本
    }

    fn compute2(&self, _c: *mut f32, _rope_ptr: *const f32) {
        // TODO: f32 版本
    }
}
