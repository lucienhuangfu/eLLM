// === compiler/mul/matmul.rs ===
#![allow(non_snake_case)]

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::common::matmul_params::MatMulParams;
use crate::kernel;
use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::MatMulTrait;

// Variable naming used in this operator:
// - input_rows / input_row_start: rows from A and C.
// - output_cols / output_col_start: columns from B_nt and C.
// - reduction_cols / reduction_col_start: the K dimension reduced by GEMM.
// - input_block_rows / output_block_cols / reduction_block_cols: macro tile sizes.
// - micro_tile_rows / micro_tile_cols: micro-kernel tile size.
// 本算子的变量命名约定：
// - input_rows / input_row_start：A 和 C 的行维度。
// - output_cols / output_col_start：B_nt 和 C 的列维度。
// - reduction_cols / reduction_col_start：GEMM 中被规约的 K 维度。
// - input_block_rows / output_block_cols / reduction_block_cols：宏块大小。
// - micro_tile_rows / micro_tile_cols：微内核 tile 大小。

#[derive(Clone)]
pub struct MatMul<T> {
    pub ptr1: ConstPtr<T>,     // Input matrix A: [input_rows, reduction_cols].
    pub ptr2: ConstPtr<T>,     // Packed source weight B_nt: [output_cols, reduction_cols].
    pub output_ptr: MutPtr<T>, // Output matrix C: [input_rows, output_cols].

    pub output_to_kv: bool,

    /// Blocking shape only: token/output/reduction macro blocks and micro tile size.
    /// 只承载分块形状：token/output/reduction 宏块和微内核 tile 大小。
    pub params: MatMulParams,
    pub _marker: PhantomData<T>,

    // Maximum runtime dimensions.
    // 运行时最大维度。
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
    pub decode_only_flag: bool,

    // Weight panels are packed in new(), so run() does not pack or allocate.
    // 权重 panel 在 new() 中提前 pack，run() 中不 pack、不分配。
    packed_b: Box<[T]>, // [reduction_panels][output_panels][reduction_block * micro_cols]
    packed_panel_stride: usize, // reduction_block_cols * micro_tile_cols
}

impl<T> MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// Create a matmul operator from already-transposed B weights.
    /// 从已经转置好的 B 权重创建 matmul operator。
    ///
    /// B is expected as B_nt[output_cols, reduction_cols].
    /// B 要求传入 B_nt[output_cols, reduction_cols]。
    pub unsafe fn new(
        ptr1: *const T,          // A[input_rows, reduction_cols]
        ptr2_b_nt_nxk: *const T, // B_nt[output_cols, reduction_cols]
        output_ptr: *mut T,      // C[input_rows, output_cols]
        output_to_kv: bool,
        params: MatMulParams, // Blocking shape only. 只表示分块形状。
        m_max: usize,
        n_max: usize,
        k_max: usize,
        decode_only_flag: bool,
    ) -> Self {
        let reduction_block_cols = params.column_step_macro.max(1);
        let micro_tile_cols = params.b_row_step_micro.max(1);
        let packed_panel_stride = reduction_block_cols * micro_tile_cols;
        let packed_b = Self::pack_b_panels(
            ptr2_b_nt_nxk,
            n_max,
            k_max,
            reduction_block_cols,
            micro_tile_cols,
        );

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
            decode_only_flag,
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
    fn pack_b_panels(
        weight_nt: *const T,
        output_cols: usize,
        reduction_cols: usize,
        reduction_block_cols: usize,
        micro_tile_cols: usize,
    ) -> Box<[T]> {
        let reduction_panel_count = reduction_cols.div_ceil(reduction_block_cols);
        let output_panel_count = output_cols.div_ceil(micro_tile_cols);
        let panel_stride = reduction_block_cols * micro_tile_cols;
        let mut packed =
            vec![T::default(); reduction_panel_count * output_panel_count * panel_stride];

        unsafe {
            for reduction_panel_index in 0..reduction_panel_count {
                let reduction_start = reduction_panel_index * reduction_block_cols;
                let reduction_cols_this =
                    (reduction_cols - reduction_start).min(reduction_block_cols);
                for output_panel_index in 0..output_panel_count {
                    let output_start = output_panel_index * micro_tile_cols;
                    let output_cols_this = (output_cols - output_start).min(micro_tile_cols);
                    let panel = packed.as_mut_ptr().add(
                        (reduction_panel_index * output_panel_count + output_panel_index)
                            * panel_stride,
                    );
                    for reduction_lane in 0..reduction_cols_this {
                        let packed_row = panel.add(reduction_lane * micro_tile_cols);
                        for output_lane in 0..output_cols_this {
                            *packed_row.add(output_lane) = *weight_nt.add(
                                (output_start + output_lane) * reduction_cols
                                    + (reduction_start + reduction_lane),
                            );
                        }
                    }
                }
            }
        }

        packed.into_boxed_slice()
    }

    #[inline(always)]
    fn packed_panel_ptr(&self, output_col_start: usize, reduction_col_start: usize) -> *const T {
        let reduction_block_cols = self.params.column_step_macro.max(1);
        let micro_tile_cols = self.params.b_row_step_micro.max(1);
        let output_panel_count = self.n_max.div_ceil(micro_tile_cols);
        let panel_index = (reduction_col_start / reduction_block_cols) * output_panel_count
            + (output_col_start / micro_tile_cols);
        unsafe {
            self.packed_b
                .as_ptr()
                .add(panel_index * self.packed_panel_stride)
        }
    }

    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let active_input_rows = if prefill_size == 0 || self.decode_only_flag {
                decode_size
            } else {
                prefill_size
            };

            let output_cols = self.n_max;
            let reduction_cols = self.k_max;

            let input_block_rows = self.params.a_row_step_macro.max(1);
            let output_block_cols = self.params.b_row_step_macro.max(1);
            let reduction_block_cols = self.params.column_step_macro.max(1);
            let micro_tile_rows = self.params.a_row_step_micro.max(1);
            let micro_tile_cols = self.params.b_row_step_micro.max(1);

            // The micro-kernel runs fixed-size row/column tiles.
            // 微内核按固定行/列 tile 计算。

            // Pad active rows to the micro tile height; padded rows are harmless extra compute.
            // 将实际行数补齐到微内核行数倍数；补齐行只是空算。
            let padded_input_rows = active_input_rows.div_ceil(micro_tile_rows) * micro_tile_rows;

            // new() must reserve enough memory for the padded rows.
            // new() 必须预留足够覆盖 padded rows 的内存。
            debug_assert!(padded_input_rows <= self.m_max);

            // Macro blocks must align with micro tiles, so run() needs no tail micro-kernel.
            // 宏块必须与微内核 tile 对齐，这样 run() 不需要额外 tail 微核。
            debug_assert!(input_block_rows % micro_tile_rows == 0);

            debug_assert!(output_cols % micro_tile_cols == 0);
            debug_assert!(reduction_cols % reduction_block_cols == 0);

            debug_assert!(thread_num >= 1);
            debug_assert!(thread_id < thread_num);

            let input_base = self.ptr1.ptr;
            let output_base = self.output_ptr.ptr;
            let input_row_stride = reduction_cols;
            let output_row_stride = output_cols;

            let input_tile_count = padded_input_rows.div_ceil(input_block_rows);
            let output_tile_count = output_cols.div_ceil(output_block_cols);
            let total_tiles = input_tile_count * output_tile_count;

            if let Some((task_begin, task_end)) = assign(total_tiles, thread_num, thread_id) {
                for task_id in task_begin..task_end {
                    let input_tile_id = task_id / output_tile_count;
                    let output_tile_id = task_id % output_tile_count;

                    let input_row_start = input_tile_id * input_block_rows;
                    let output_col_start = output_tile_id * output_block_cols;

                    let input_rows_in_block =
                        (padded_input_rows - input_row_start).min(input_block_rows);
                    let output_cols_in_block =
                        (output_cols - output_col_start).min(output_block_cols);

                    debug_assert!(
                        input_rows_in_block % micro_tile_rows == 0
                            && output_cols_in_block % micro_tile_cols == 0
                    );

                    let mut reduction_col_start = 0;
                    while reduction_col_start < reduction_cols {
                        let mut output_col_offset = 0;
                        while output_col_offset < output_cols_in_block {
                            let weight_panel_ptr = self.packed_panel_ptr(
                                output_col_start + output_col_offset,
                                reduction_col_start,
                            );

                            let mut input_row_offset = 0;
                            while input_row_offset < input_rows_in_block {
                                let input_tile = input_base.add(
                                    (input_row_start + input_row_offset) * input_row_stride
                                        + reduction_col_start,
                                );
                                let output_tile = output_base.add(
                                    (input_row_start + input_row_offset) * output_row_stride
                                        + (output_col_start + output_col_offset),
                                );

                                self.compute(input_tile, weight_panel_ptr, output_tile);
                                input_row_offset += micro_tile_rows;
                            }

                            output_col_offset += micro_tile_cols;
                        }
                        reduction_col_start += reduction_block_cols;
                    }
                }
            }
        }
    }
}

/* ------------------ compute/compute2 default implementations ------------------ */
/* ------------------ compute/compute2 默认实现 ------------------ */

impl<T> MatMulTrait<T> for MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max, // input row stride / 输入行距, lda = K
            b_row_step_macro: self.n_max, // output row stride / 输出行距, ldc = N
            column_step_macro: self.params.column_step_macro, // reduction block / 规约块
            a_row_step_micro: self.params.a_row_step_micro, // micro rows / 微内核行数
            b_row_step_micro: self.params.b_row_step_micro, // micro cols / 微内核列数
        };

        kernel::scalar::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
    }

    default fn compute2(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
        length: usize,
    ) {
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
            kernel::x86_64::f16_512::matmul_block::matmul_block(
                input_ptr1,
                input_ptr2,
                output_ptr,
                &call_param,
            );
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::scalar::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
    }

    fn compute2(
        &self,
        input_ptr1: *const f16,
        input_ptr2: *const f16,
        output_ptr: *mut f16,
        length: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::dot_product::dot_product(
                input_ptr1, input_ptr2, output_ptr, length,
            );
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::scalar::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn avail_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
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

    #[test]
    fn test_matmul_runner_f32_alignment_fp32_small_qwen_style() {
        use approx::assert_abs_diff_eq;

        const M_RUN: usize = 4;
        const M_MAX: usize = 6; // pad to MR=3
        const K: usize = 64;
        const N: usize = 32;

        let thread_num = 1;

        let mut a = vec![0.0f32; M_MAX * K];
        let mut b_nt = vec![0.0f32; N * K];
        let mut c = vec![0.0f32; M_MAX * N];

        // Synthetic deterministic inputs: easy to debug and stable across runs.
        for i in 0..M_RUN {
            for kk in 0..K {
                a[i * K + kk] = 0.01f32 * (i as f32) + 0.001f32 * (kk as f32);
            }
        }

        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = 0.02f32 * (kk as f32) + 0.003f32 * (j as f32);
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 32,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let matmul = unsafe {
            MatMul::<f32>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                c.as_mut_ptr(),
                false,
                params,
                M_MAX,
                N,
                K,
                false,
            )
        };

        for tid in 0..thread_num {
            matmul.run(M_RUN, 0, thread_num, tid);
        }

        for i in 0..M_RUN {
            for j in 0..N {
                let mut sum = 0.0f32;
                for kk in 0..K {
                    sum += a[i * K + kk] * b_nt[j * K + kk];
                }

                let got = c[i * N + j];
                assert_abs_diff_eq!(got, sum, epsilon = 1e-5);
            }
        }

        for i in M_RUN..M_MAX {
            for j in 0..N {
                assert_abs_diff_eq!(c[i * N + j], 0.0f32, epsilon = 1e-5);
            }
        }
    }
}
