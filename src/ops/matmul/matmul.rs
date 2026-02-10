// === ops/mul/matmul.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use crate::kernel;
use crate::ops::assign::assign;
use crate::ops::traits::mul_trait::MatMulTrait;

#[derive(Clone)]
pub struct MatMul<T> {
    pub ptr1: ConstPtr<T>,     // A[M×K]
    pub ptr2: ConstPtr<T>,     // B_nt[N×K]（按行连续）
    pub output_ptr: MutPtr<T>, // C[M×N]

    pub output_to_kv: bool,

    /// 仅承载 step 形状（MB/NB/KC/MR/NR）
    pub params: MatMulParams,
    pub _marker: PhantomData<T>,

    // 最大维度
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,

    // 线程私有 KC×NR 面板池：连续大块，按 thread_id 切片
    // 布局：[threads][kc*nr]
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize, // = kc * nr
}

impl<T> MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// new():
    /// 1) 不再做 B 转置：直接接收 B_nt[N×K]
    /// 2) 根据 CPU 可用并行度预分配面板池（只在 new() 里使用线程数，不进结构体字段）
    pub unsafe fn new(
        ptr1: *const T,          // A[M×K]
        ptr2_b_nt_nxk: *const T, // ✅ B_nt[N×K]（按行连续）
        output_ptr: *mut T,      // C[M×N]
        output_to_kv: bool,
        params: MatMulParams, // 仅 step 形状
        m_max: usize,
        n_max: usize,
        k_max: usize,
        decode_only_flag: bool,
    ) -> Self {
        // === (1) 不再构造期转置：ptr2 直接引用传入的 B_nt[N×K] ===
        let b_nt_base = ptr2_b_nt_nxk;

        // === (2) 预分配 panel pool：线程数来自 CPU 并行度 ===
        let kc = params.column_step_macro.max(1);
        let nr = params.b_row_step_micro.max(1);
        let b_panel_stride_elems = kc * nr;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let pool_len = threads * b_panel_stride_elems;
        let b_panel_pool: Vec<T> = vec![T::default(); pool_len];

        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: b_nt_base },
            output_ptr: MutPtr { ptr: output_ptr },
            output_to_kv,
            params,
            _marker: PhantomData,
            m_max,
            n_max,
            k_max,
            b_panel_pool: b_panel_pool.into_boxed_slice(),
            b_panel_stride_elems,
        }
    }

    /// 当前 pool 支持的线程数（由 pool_len / stride 推导）
    #[inline(always)]
    pub fn panel_threads(&self) -> usize {
        if self.b_panel_stride_elems == 0 {
            0
        } else {
            self.b_panel_pool.len() / self.b_panel_stride_elems
        }
    }

    /// 取得本线程的 KC×NR 面板指针（不分配）
    #[inline(always)]
    pub fn thread_b_panel_ptr(&self, thread_id: usize) -> *mut T {
        unsafe {
            self.b_panel_pool
                .as_ptr()
                .add(thread_id * self.b_panel_stride_elems) as *mut T
        }
    }

    pub fn run(&self, prefill_size: usize, _decode_size: usize, thread_num: usize, thread_id: usize) {

    unsafe {
        let m_run = prefill_size;

        let n = self.n_max;
        let k = self.k_max;

        let mb = self.params.a_row_step_macro.max(1);
        let nb = self.params.b_row_step_macro.max(1);
        let kc = self.params.column_step_macro.max(1);
        let mr = self.params.a_row_step_micro.max(1);
        let nr = self.params.b_row_step_micro.max(1);

        // === 固定微核假设：MR=3, NR=32 时你也可以留着，但这里保留通用写法 ===

        // 关键：run 时把 M 向上 pad 到 MR 的倍数（空算），用于跑固定 MR 微核
        let m_pad = ((m_run + mr - 1) / mr) * mr;

        // new() 预留必须覆盖到 m_pad（你说你能保证，这里用断言锁死）
        debug_assert!(m_pad <= self.m_max);

        // 你当前的块内循环要求 m_blk % mr == 0，
        // 为了保证这一点且不引入 tile 内 tail 逻辑，要求 MB 是 MR 的倍数
        debug_assert!(mb % mr == 0);

        // 其他对齐断言：你现在 pack/微核都依赖这些
        debug_assert!(n % nr == 0);
        debug_assert!(k % kc == 0);

        // 线程数只要不超过 pool 支持即可
        let max_threads = self.panel_threads();
        debug_assert!(thread_num >= 1);
        debug_assert!(thread_id < thread_num);
        debug_assert!(thread_num <= max_threads);

        let a_base = self.ptr1.ptr;
        let c_base = self.output_ptr.ptr;
        let lda = k;
        let ldc = n;

        let b_nt_ptr = self.ptr2.ptr; // N×K row-major
        let ldb_row = k;

        let b_panel_ptr = self.thread_b_panel_ptr(thread_id);

        #[inline(always)]
        unsafe fn pack_b_panel<T: Copy>(
            b_nt: *const T,
            ldb_row: usize,
            n0: usize,
            k0: usize,
            kc: usize,
            nr: usize,
            out: *mut T,
        ) {
            for p in 0..kc {
                let src_col = k0 + p;
                let dst_row = out.add(p * nr);
                for lane in 0..nr {
                    let j = n0 + lane;
                    let src = b_nt.add(j * ldb_row + src_col);
                    *dst_row.add(lane) = *src;
                }
            }
        }

        // 这里开始：所有 tile 切分按 m_pad 而不是 m_run
        let tiles_m = (m_pad + mb - 1) / mb;
        let tiles_n = (n + nb - 1) / nb;
        let tiles = tiles_m * tiles_n;

        if let Some((tb, te)) = assign(tiles, thread_num, thread_id) {
            for t in tb..te {
                let tm = t / tiles_n;
                let tn = t % tiles_n;

                let m0 = tm * mb;
                let n0 = tn * nb;

                // 注意：m_blk 基于 m_pad
                let m_blk = (m_pad - m0).min(mb);
                let n_blk = (n - n0).min(nb);

                debug_assert!(m_blk % mr == 0 && n_blk % nr == 0);

                let mut k0 = 0;
                while k0 < k {
                    let mut nt = 0;
                    while nt < n_blk {
                        pack_b_panel::<T>(b_nt_ptr, ldb_row, n0 + nt, k0, kc, nr, b_panel_ptr);

                        let mut mi = 0;
                        while mi < m_blk {
                            let a_tile = a_base.add((m0 + mi) * lda + k0);
                            let c_tile = c_base.add((m0 + mi) * ldc + (n0 + nt));

                            self.compute(a_tile, b_panel_ptr as *const T, c_tile);
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

/* ------------------ compute/compute2：保持你的调用风格 ------------------ */

impl<T> MatMulTrait<T> for MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max,                     // lda = K
            b_row_step_macro: self.n_max,                     // ldc = N
            column_step_macro: self.params.column_step_macro, // kc
            a_row_step_micro: self.params.a_row_step_micro,   // mr
            b_row_step_micro: self.params.b_row_step_micro,   // nr
        };

        kernel::scalar::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
    }

    default fn compute2(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T, length: usize) {
        kernel::scalar::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

impl MatMulTrait<f16> for MatMul<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max,
            b_row_step_macro: self.n_max,
            column_step_macro: self.params.column_step_macro,
            a_row_step_micro: self.params.a_row_step_micro,
            b_row_step_micro: self.params.b_row_step_micro,
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::scalar::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
    }

    fn compute2(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16, length: usize) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::scalar::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

impl MatMulTrait<f32> for MatMul<f32> {
    fn compute(&self, _a: *const f32, _b: *const f32, _c: *mut f32) {
        /* TODO */
    }
    fn compute2(&self, _a: *const f32, _b: *const f32, _c: *mut f32, _length: usize) {
        /* TODO */
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn avail_threads() -> usize {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    }

    #[test]
    fn test_matmul_runner_f16_3x64x32() {
        const M: usize = 3;
        const K: usize = 64;
        const N: usize = 32;

        let thread_num = avail_threads().min(8); // 测试别太大

        let mut a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K]; // ✅ B_nt[N×K]
        let mut c = vec![0.0f16; M * N];

        for i in 0..M {
            for kk in 0..K {
                let v = 0.01f32 * (i as f32) + 0.001f32 * (kk as f32);
                a[i * K + kk] = v as f16;
            }
        }

        // ✅ 填充 B_nt：按 N×K row-major
        for j in 0..N {
            for kk in 0..K {
                let v = 0.02f32 * (kk as f32) + 0.003f32 * (j as f32);
                b_nt[j * K + kk] = v as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 3,   // MB
            b_row_step_macro: 32,  // NB
            column_step_macro: 64, // KC
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        let matmul = unsafe {
            MatMul::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅ 传入 B_nt[N×K]
                c.as_mut_ptr(),
                false,
                params,
                M,
                N,
                K,
                false,
            )
        };

        // 顺序模拟多线程调用
        for tid in 0..thread_num {
            matmul.run(M, 0, thread_num, tid);
        }

        for i in 0..M {
            for j in 0..N {
                let mut sum = 0.0f32;
                for kk in 0..K {
                    // ✅ reference：A[M×K] × B[K×N]，但我们存的是 B_nt[N×K]
                    // B[k][j] == B_nt[j][k]
                    sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
                }
                let got = c[i * N + j] as f32;
                assert_abs_diff_eq!(got, sum, epsilon = 1e-1);
            }
        }
    }

    #[test]
    fn test_matmul_runner_f16_144x2048x2048() {
        const M: usize = 144;
        const K: usize = 2048;
        const N: usize = 2048;

        let thread_num = avail_threads().min(16);

        let mut a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K]; // ✅ B_nt[N×K]
        let mut c = vec![0.0f16; M * N];

        for i in 0..M {
            for k in 0..K {
                let val = ((i + k) % 7) as f32 * 0.01;
                a[i * K + k] = val as f16;
            }
        }

        // ✅ 填充 B_nt：按 N×K row-major
        for j in 0..N {
            for k in 0..K {
                let val = ((k + j) % 11) as f32 * 0.01;
                b_nt[j * K + k] = val as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 24,  // MB
            b_row_step_macro: 128, // NB
            column_step_macro: 64, // KC
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        let matmul = unsafe {
            MatMul::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅ 传入 B_nt[N×K]
                c.as_mut_ptr(),
                false,
                params,
                M,
                N,
                K,
                false,
            )
        };

        for tid in 0..thread_num {
            matmul.run(M, 0, thread_num, tid);
        }

        for i in 0..M {
            for j in 0..N {
                let mut sum = 0.0f32;
                for kk in 0..K {
                    sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
                }
                let got = c[i * N + j] as f32;
                assert_abs_diff_eq!(got, sum, epsilon = 5e-1);
            }
        }
    }
    #[test]
    fn test_matmul_runner_f16_batch7_pad_to9() {
        const M_RUN: usize = 7;
        const MR: usize = 3;
        const M_MAX: usize = 9; // ceil_div(7,3)*3=9
        const K: usize = 64;
        const N: usize = 32;

        let thread_num = 1; // 单线程先验证逻辑，避免并发干扰

        // A/C 按 M_MAX 分配，前 M_RUN 行填数据，pad 行填 0
        let mut a = vec![0.0f16; M_MAX * K];
        let mut b_nt = vec![0.0f16; N * K]; // B_nt[N×K]
        let mut c = vec![0.0f16; M_MAX * N];

        // 填 A：前 7 行有值，剩下两行保持 0
        for i in 0..M_RUN {
            for kk in 0..K {
                let v = 0.01f32 * (i as f32) + 0.001f32 * (kk as f32);
                a[i * K + kk] = v as f16;
            }
        }

        // 填 B_nt：按 N×K row-major
        for j in 0..N {
            for kk in 0..K {
                let v = 0.02f32 * (kk as f32) + 0.003f32 * (j as f32);
                b_nt[j * K + kk] = v as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 6,   // MB（是 MR 的倍数）
            b_row_step_macro: 32,  // NB
            column_step_macro: 64, // KC
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        let matmul = unsafe {
            MatMul::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                c.as_mut_ptr(),
                false,
                params,
                M_MAX, // m_max=9
                N,
                K,
                false,
            )
        };

        // batch_size 传 7（不是 3 的倍数），内部会 pad 到 9
        for tid in 0..thread_num {
            matmul.run(M_RUN, 0, thread_num, tid);
        }

        // 只检查前 7 行（真实 batch），pad 行不检查
        for i in 0..M_RUN {
            for j in 0..N {
                let mut sum = 0.0f32;
                for kk in 0..K {
                    sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
                }
                let got = c[i * N + j] as f32;
                approx::assert_abs_diff_eq!(got, sum, epsilon = 1e-1);
            }
        }

        // 可选：pad 行如果 A pad 行为 0，那么 C pad 行应该仍为 0（这里只是额外 sanity）
        for i in M_RUN..M_MAX {
            for j in 0..N {
                let got = c[i * N + j] as f32;
                approx::assert_abs_diff_eq!(got, 0.0, epsilon = 1e-1);
            }
        }

        // 额外：确认 MR 固定=3 的前提没被破坏
        assert_eq!(MR, 3);
    }
}


