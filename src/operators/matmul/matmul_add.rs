// === compiler/mul/matmul_add.rs ===
#![allow(non_snake_case)]
#![allow(unused_variables)] // Keep trait-compatible unused parameters. 保留 trait 兼容的未使用参数。

use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::common::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use crate::kernel;
use crate::operators::assign::assign;
use crate::operators::traits::MatMulAddTrait;

// Variable naming used in this operator:
// - input_rows / input_row_start: rows from A, residual, and C.
// - output_cols / output_col_start: columns from B_nt, residual, and C.
// - reduction_cols / reduction_col_start: the K dimension reduced by GEMM.
// - input_block_rows / output_block_cols / reduction_block_cols: macro tile sizes.
// - micro_tile_rows / micro_tile_cols: micro-kernel tile size.
// 本算子的变量命名约定：
// - input_rows / input_row_start：A、residual 和 C 的行维度。
// - output_cols / output_col_start：B_nt、residual 和 C 的列维度。
// - reduction_cols / reduction_col_start：GEMM 中被规约的 K 维度。
// - input_block_rows / output_block_cols / reduction_block_cols：宏块大小。
// - micro_tile_rows / micro_tile_cols：微内核 tile 大小。

#[derive(Clone)]
pub struct MatMulAdd<T> {
    pub ptr1: ConstPtr<T>,     // Input matrix A: [input_rows, reduction_cols].
    pub ptr2: ConstPtr<T>,     // Weight matrix B_nt: [output_cols, reduction_cols].
    pub ptr3: ConstPtr<T>,     // Residual matrix: [input_rows, output_cols].
    pub output_ptr: MutPtr<T>, // Output matrix C: [input_rows, output_cols].

    /// Blocking shape only: token/output/reduction macro blocks and micro tile size.
    /// 只承载分块形状：token/output/reduction 宏块和微内核 tile 大小。
    pub params: MatMulParams,
    pub _marker: PhantomData<T>,

    // Maximum runtime dimensions, matching MatMul.
    // 运行时最大维度，与 MatMul 保持一致。
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
    pub decode_only_flag: bool,

    // Weight panels are packed in new(), so run() only reuses prepared memory.
    // 权重 panel 在 new() 中提前 pack，run() 中只复用已准备好的内存。
    packed_b: Box<[T]>, // [reduction_panels][output_panels][reduction_block * micro_cols]
    packed_panel_stride: usize, // reduction_block_cols * micro_tile_cols
}

impl<T> MatMulAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// Create a residual-add matmul operator from already-transposed B weights.
    /// 从已经转置好的 B 权重创建带 residual add 的 matmul operator。
    pub unsafe fn new(
        ptr1: *const T,          // A[input_rows, reduction_cols]
        ptr2_b_nt_nxk: *const T, // B_nt[output_cols, reduction_cols]
        ptr3_residual: *const T, // residual[input_rows, output_cols]
        output_ptr: *mut T,      // C[input_rows, output_cols]
        params: MatMulParams,
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
            ptr3: ConstPtr { ptr: ptr3_residual },
            output_ptr: MutPtr { ptr: output_ptr },
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

    /// Run: copy residual into output first, then accumulate output += A * B.
    /// 执行：先把 residual 覆盖到 output，然后做 output += A * B。
    ///
    /// The run signature stays aligned with Operator.
    /// run 签名保持与 Operator 统一。
    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let active_input_rows = if self.decode_only_flag {
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

            // Pad active rows to the micro tile height; padded rows are harmless extra compute.
            // 将实际行数补齐到微内核行数倍数；补齐行只是空算。
            let padded_input_rows = active_input_rows.div_ceil(micro_tile_rows) * micro_tile_rows;
            debug_assert!(padded_input_rows <= self.m_max);

            debug_assert!(input_block_rows % micro_tile_rows == 0);
            debug_assert!(output_cols % micro_tile_cols == 0);
            debug_assert!(reduction_cols % reduction_block_cols == 0);
            debug_assert!(thread_num >= 1);
            debug_assert!(thread_id < thread_num);

            let input_base = self.ptr1.ptr;
            let residual_base = self.ptr3.ptr;
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

                    // Copy residual tile into output before accumulation.
                    // 累加前先把 residual tile 拷贝到 output。
                    let mut output_col_offset = 0;
                    while output_col_offset < output_cols_in_block {
                        let mut input_row_offset = 0;
                        while input_row_offset < input_rows_in_block {
                            let residual_tile = residual_base.add(
                                (input_row_start + input_row_offset) * output_row_stride
                                    + (output_col_start + output_col_offset),
                            );
                            let output_tile = output_base.add(
                                (input_row_start + input_row_offset) * output_row_stride
                                    + (output_col_start + output_col_offset),
                            );

                            for row_in_tile in 0..micro_tile_rows {
                                let residual_row =
                                    residual_tile.add(row_in_tile * output_row_stride);
                                let output_row = output_tile.add(row_in_tile * output_row_stride);
                                std::ptr::copy_nonoverlapping(
                                    residual_row,
                                    output_row,
                                    micro_tile_cols,
                                );
                            }

                            input_row_offset += micro_tile_rows;
                        }
                        output_col_offset += micro_tile_cols;
                    }

                    // Accumulate output += input * weight.
                    // 累加 output += input * weight。
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

                                self.compute(
                                    input_tile,
                                    weight_panel_ptr,
                                    std::ptr::null(),
                                    output_tile,
                                );

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

/* ------------------ compute default implementation ------------------ */
/* ------------------ compute 默认实现 ------------------ */

impl<T> MatMulAddTrait<T> for MatMulAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        _input_ptr3: *const T,
        output_ptr: *mut T,
    ) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max, // input row stride / 输入行距, lda = K
            b_row_step_macro: self.n_max, // output row stride / 输出行距, ldc = N
            column_step_macro: self.params.column_step_macro, // reduction block / 规约块
            a_row_step_micro: self.params.a_row_step_micro, // micro rows / 微内核行数
            b_row_step_micro: self.params.b_row_step_micro, // micro cols / 微内核列数
        };
        kernel::scalar::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &call_param);
    }
}

// f16 specialization: AVX-512 FP16 accumulation micro-kernel.
// f16 特化：AVX-512 FP16 累加微内核。
impl MatMulAddTrait<f16> for MatMulAdd<f16> {
    fn compute(
        &self,
        input_ptr1: *const f16,
        input_ptr2: *const f16,
        _input_ptr3: *const f16,
        output_ptr: *mut f16,
    ) {
        let call_param = MatMulParams {
            a_row_step_macro: self.k_max, // input row stride / 输入行距, lda = K
            b_row_step_macro: self.n_max, // output row stride / 输出行距, ldc = N
            column_step_macro: self.params.column_step_macro, // reduction block / 规约块
            a_row_step_micro: self.params.a_row_step_micro, // micro rows / 微内核行数
            b_row_step_micro: self.params.b_row_step_micro, // micro cols / 微内核列数
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
    fn test_matmul_add_new_packs_b_panels_f16() {
        const K: usize = 8;
        const N: usize = 6;
        const M: usize = 3;

        let a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K];
        let residual = vec![0.0f16; M * N];
        let mut c = vec![0.0f16; M * N];

        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = (100 * j + kk) as f32 as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 4,
            a_row_step_micro: 3,
            b_row_step_micro: 2,
        };

        let runner = unsafe {
            MatMulAdd::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                residual.as_ptr(),
                c.as_mut_ptr(),
                params,
                M,
                N,
                K,
                false,
            )
        };

        let panel_00 = unsafe { std::slice::from_raw_parts(runner.packed_panel_ptr(0, 0), 8) };
        let expected_00 = [0.0, 100.0, 1.0, 101.0, 2.0, 102.0, 3.0, 103.0];
        for (got, expected) in panel_00.iter().zip(expected_00) {
            assert_abs_diff_eq!(*got as f32, expected, epsilon = 0.0);
        }

        let panel_24 = unsafe { std::slice::from_raw_parts(runner.packed_panel_ptr(2, 4), 8) };
        let expected_24 = [204.0, 304.0, 205.0, 305.0, 206.0, 306.0, 207.0, 307.0];
        for (got, expected) in panel_24.iter().zip(expected_24) {
            assert_abs_diff_eq!(*got as f32, expected, epsilon = 0.0);
        }
    }

    #[test]
    fn test_matmul_add_panel_threads_available_f16() {
        const M: usize = 3;
        const K: usize = 64;
        const N: usize = 32;

        let a = vec![0.0f16; M * K];
        let b_nt = vec![0.0f16; N * K];
        let residual = vec![0.0f16; M * N];
        let mut c = vec![0.0f16; M * N];

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let runner = unsafe {
            MatMulAdd::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                residual.as_ptr(),
                c.as_mut_ptr(),
                params,
                M,
                N,
                K,
                false,
            )
        };

        assert!(runner.panel_threads() >= 1);
    }

    #[test]
    fn test_matmul_add_runner_f16_nt_6x64x32() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            // 这只是 runner test；不强制要求 avx512 才能跑也行
        }

        const M: usize = 6;
        const K: usize = 64;
        const N: usize = 32;

        let thread_num = avail_threads().min(8).max(1);

        let mut a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K];
        let mut residual = vec![0.0f16; M * N];
        let mut c = vec![0.0f16; M * N];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (0.01f32 * (i as f32) + 0.001f32 * (kk as f32)) as f16;
            }
        }
        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = (0.02f32 * (kk as f32) + 0.003f32 * (j as f32)) as f16;
            }
        }
        for i in 0..M {
            for j in 0..N {
                residual[i * N + j] = (0.05f32 * (i as f32) + 0.0007f32 * (j as f32)) as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: M,  // MB
            b_row_step_macro: N,  // NB
            column_step_macro: K, // KC
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let runner = unsafe {
            MatMulAdd::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(), // ✅ NT
                residual.as_ptr(),
                c.as_mut_ptr(),
                params,
                M,
                N,
                K,
                false,
            )
        };

        for tid in 0..thread_num {
            runner.run(M, 0, thread_num, tid);
        }

        for i in 0..M {
            for j in 0..N {
                let mut sum = residual[i * N + j] as f32;
                for kk in 0..K {
                    sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
                }
                let got = c[i * N + j] as f32;
                assert_abs_diff_eq!(got, sum, epsilon = 2e-1);
            }
        }
    }
    #[test]
    fn test_matmul_add_runner_f16_nt_batch7_pad_to9() {
        const M_RUN: usize = 7;
        const M_MAX: usize = 9; // ceil_div(7,3)*3
        const K: usize = 64;
        const N: usize = 32;

        let thread_num = avail_threads().min(8).max(1);

        let mut a = vec![0.0f16; M_MAX * K];
        let mut b_nt = vec![0.0f16; N * K];
        let mut residual = vec![0.0f16; M_MAX * N];
        let mut c = vec![0.0f16; M_MAX * N];

        for i in 0..M_RUN {
            for kk in 0..K {
                a[i * K + kk] = (0.01f32 * (i as f32) + 0.001f32 * (kk as f32)) as f16;
            }
        }

        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = (0.02f32 * (kk as f32) + 0.003f32 * (j as f32)) as f16;
            }
        }

        for i in 0..M_RUN {
            for j in 0..N {
                residual[i * N + j] = (0.05f32 * (i as f32) + 0.0007f32 * (j as f32)) as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 32,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let runner = unsafe {
            MatMulAdd::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                residual.as_ptr(),
                c.as_mut_ptr(),
                params,
                M_MAX,
                N,
                K,
                false,
            )
        };

        let used = thread_num.min(runner.panel_threads()).max(1);
        for tid in 0..used {
            runner.run(M_RUN, 0, used, tid);
        }

        for i in 0..M_RUN {
            for j in 0..N {
                let mut sum = residual[i * N + j] as f32;
                for kk in 0..K {
                    sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
                }
                let got = c[i * N + j] as f32;
                assert_abs_diff_eq!(got, sum, epsilon = 2e-1);
            }
        }

        for i in M_RUN..M_MAX {
            for j in 0..N {
                let got = c[i * N + j] as f32;
                assert_abs_diff_eq!(got, 0.0, epsilon = 1e-1);
            }
        }
    }
}
