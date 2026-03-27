// === compiler/mul/matmul.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::MatMulTrait;

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

    packed_b: Box<[T]>,         // [panels_k][panels_n][kc*nr]
    packed_panel_stride: usize, // = kc * nr
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
    ) -> Self {
        let kc = params.column_step_macro.max(1);
        let nr = params.b_row_step_micro.max(1);
        let packed_panel_stride = kc * nr;
        let packed_b = Self::pack_b_panels(ptr2_b_nt_nxk, n_max, k_max, kc, nr);

        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: ptr2_b_nt_nxk },
            output_ptr: MutPtr { ptr: output_ptr },
            output_to_kv,
            params,
            _marker: PhantomData,
            m_max,
            n_max,
            k_max,
            packed_b,
            packed_panel_stride,
        }
    }

    #[inline(always)]
    pub fn panel_threads(&self) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    #[inline(always)]
    fn pack_b_panels(b_nt: *const T, n: usize, k: usize, kc: usize, nr: usize) -> Box<[T]> {
        let panels_k = k.div_ceil(kc);
        let panels_n = n.div_ceil(nr);
        let panel_stride = kc * nr;
        let mut packed = vec![T::default(); panels_k * panels_n * panel_stride];

        unsafe {
            for kb in 0..panels_k {
                let k0 = kb * kc;
                let kc_cur = (k - k0).min(kc);
                for nb in 0..panels_n {
                    let n0 = nb * nr;
                    let nr_cur = (n - n0).min(nr);
                    let panel = packed.as_mut_ptr().add((kb * panels_n + nb) * panel_stride);
                    for p in 0..kc_cur {
                        let dst_row = panel.add(p * nr);
                        for lane in 0..nr_cur {
                            *dst_row.add(lane) = *b_nt.add((n0 + lane) * k + (k0 + p));
                        }
                    }
                }
            }
        }

        packed.into_boxed_slice()
    }

    #[inline(always)]
    fn packed_panel_ptr(&self, n0: usize, k0: usize) -> *const T {
        let kc = self.params.column_step_macro.max(1);
        let nr = self.params.b_row_step_micro.max(1);
        let panels_n = self.n_max.div_ceil(nr);
        let panel_idx = (k0 / kc) * panels_n + (n0 / nr);
        unsafe {
            self.packed_b
                .as_ptr()
                .add(panel_idx * self.packed_panel_stride)
        }
    }

    #[inline(always)]
    fn run_batch_size_one_packed(&self, cpu_num: usize, thread_id: usize) {
        let n = self.n_max;
        let k = self.k_max;
        let nb = self.params.b_row_step_macro.max(1);
        let kc = self.params.column_step_macro.max(1);
        let nr = self.params.b_row_step_micro.max(1);

        debug_assert!(n % nr == 0);
        debug_assert!(k % kc == 0);
        debug_assert!(nb % nr == 0);
        debug_assert!(cpu_num >= 1);
        debug_assert!(thread_id < cpu_num);

        unsafe {
            let a_base = self.ptr1.ptr;
            let c_base = self.output_ptr.ptr;
            let tiles_n = n.div_ceil(nb);

            if let Some((tb, te)) = assign(tiles_n, cpu_num, thread_id) {
                for tn in tb..te {
                    let n0 = tn * nb;
                    let n_blk = (n - n0).min(nb);
                    debug_assert!(n_blk % nr == 0);

                    let mut k0 = 0;
                    while k0 < k {
                        let a_tile = a_base.add(k0);

                        let mut nt = 0;
                        while nt < n_blk {
                            let b_panel_ptr = self.packed_panel_ptr(n0 + nt, k0);
                            let c_tile = c_base.add(n0 + nt);
                            self.compute_row(a_tile, b_panel_ptr, c_tile);
                            nt += nr;
                        }

                        k0 += kc;
                    }
                }
            }
        }
    }

    pub fn run(
    &self,
    position_index: usize,
    position_interval: usize,
    batch_size: usize, // = M_run（可能不是 3 的倍数）
    cpu_num: usize,
    thread_id: usize,
) {
    let _ = position_index;
    let _ = position_interval;

    unsafe {
        let m_run = batch_size;

        if m_run == 1 {
            self.run_batch_size_one_packed(cpu_num, thread_id);
            return;
        }

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

        debug_assert!(cpu_num >= 1);
        debug_assert!(thread_id < cpu_num);

        let a_base = self.ptr1.ptr;
        let c_base = self.output_ptr.ptr;
        let lda = k;
        let ldc = n;

        // 这里开始：所有 tile 切分按 m_pad 而不是 m_run
        let tiles_m = (m_pad + mb - 1) / mb;
        let tiles_n = (n + nb - 1) / nb;
        let tiles = tiles_m * tiles_n;

        if let Some((tb, te)) = assign(tiles, cpu_num, thread_id) {
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
                        let b_panel_ptr = self.packed_panel_ptr(n0 + nt, k0);

                        let mut mi = 0;
                        while mi < m_blk {
                            let a_tile = a_base.add((m0 + mi) * lda + k0);
                            let c_tile = c_base.add((m0 + mi) * ldc + (n0 + nt));

                            self.compute(a_tile, b_panel_ptr, c_tile);
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

        kernel::generic::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
    }

    default fn compute_row(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max,
            b_row_step_macro: self.n_max,
            column_step_macro: self.params.column_step_macro,
            a_row_step_micro: 1,
            b_row_step_micro: self.params.b_row_step_micro,
        };

        kernel::generic::matmul_block::matmul_block_one_row(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &call_param,
        );
    }

    default fn compute2(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T, length: usize) {
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
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
        kernel::generic::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
    }

    fn compute_row(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max,
            b_row_step_macro: self.n_max,
            column_step_macro: self.params.column_step_macro,
            a_row_step_micro: 1,
            b_row_step_micro: self.params.b_row_step_micro,
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::matmul_block::matmul_block_one_row(
                input_ptr1,
                input_ptr2,
                output_ptr,
                &call_param,
            );
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::matmul_block::matmul_block_one_row(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &call_param,
        );
    }

    fn compute2(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16, length: usize) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

impl MatMulTrait<f32> for MatMul<f32> {
    fn compute(&self, _a: *const f32, _b: *const f32, _c: *mut f32) {
        /* TODO */
    }
    fn compute_row(&self, _a: *const f32, _b: *const f32, _c: *mut f32) {
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
            )
        };

        // 顺序模拟多线程调用
        for tid in 0..thread_num {
            matmul.run(0, 1, M, thread_num, tid);
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
            )
        };

        for tid in 0..thread_num {
            matmul.run(0, 1, M, thread_num, tid);
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
    fn test_matmul_runner_f16_batch1_direct_path() {
        const M_RUN: usize = 1;
        const M_MAX: usize = 3;
        const K: usize = 64;
        const N: usize = 96;

        let thread_num = avail_threads().min(8);

        let mut a = vec![0.0f16; M_MAX * K];
        let mut b_nt = vec![0.0f16; N * K];
        let mut c = vec![0.0f16; M_MAX * N];

        for kk in 0..K {
            a[kk] = (((kk * 13 + 5) % 29) as f32 * 0.05) as f16;
        }

        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = (((j * 7 + kk * 11) % 31) as f32 * 0.03) as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let matmul = unsafe {
            MatMul::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                c.as_mut_ptr(),
                false,
                params,
                M_MAX,
                N,
                K,
            )
        };

        for tid in 0..thread_num {
            matmul.run(0, 1, M_RUN, thread_num, tid);
        }

        for j in 0..N {
            let mut sum = 0.0f32;
            for kk in 0..K {
                sum += (a[kk] as f32) * (b_nt[j * K + kk] as f32);
            }
            assert_abs_diff_eq!(c[j] as f32, sum, epsilon = 1e-1);
        }

        for i in 1..M_MAX {
            for j in 0..N {
                assert_abs_diff_eq!(c[i * N + j] as f32, 0.0, epsilon = 1e-3);
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
        )
    };

    // batch_size 传 7（不是 3 的倍数），内部会 pad 到 9
    for tid in 0..thread_num {
        matmul.run(0, 1, M_RUN, thread_num, tid);
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
