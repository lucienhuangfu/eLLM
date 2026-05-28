use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::kernel;
use crate::kernel::common::matmul_params::MatMulParams;
use crate::num_traits::Sigmoid;
use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};

// Variable naming used in this operator:
// - input_rows / input_row_start: rows from the input matrix and sigmoid output.
// - output_cols / output_col_start: router/expert columns produced by the gate projection.
// - reduction_cols: the K dimension reduced by the matmul.
// - input_block_rows / output_block_cols: scalar block-kernel macro tile sizes.
// - micro_tile_rows: scalar block-kernel row tile size.
// 本算子的变量命名约定：
// - input_rows / input_row_start：输入矩阵和 sigmoid 输出的行维度。
// - output_cols / output_col_start：gate 投影产生的 router/expert 列。
// - reduction_cols：matmul 中被规约的 K 维度。
// - input_block_rows / output_block_cols：scalar block kernel 的宏块大小。
// - micro_tile_rows：scalar block kernel 的行方向微 tile 大小。

#[derive(Clone)]
pub struct MatMulSigmoid<T> {
    pub ptr1: ConstPtr<T>,     // Input matrix: [input_rows, reduction_cols].
    pub ptr2: ConstPtr<T>,     // Gate weight: [output_cols, reduction_cols].
    pub output_ptr: MutPtr<T>, // Sigmoid output: [input_rows, output_cols].
    pub params: MatMulParams,
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
    pub _marker: PhantomData<T>,
    // Per-thread scratch buffers reused by run().
    // 每线程 scratch buffer，run() 中复用。
    b_panel_pool: Box<[T]>,
    b_panel_stride_elems: usize,
    acc_pool: Box<[T]>,
    acc_stride_elems: usize,
    bias_ptr: Option<ConstPtr<T>>,
    use_routing_bias: bool,
    decode_only_flag: bool,
}

impl<T> MatMulSigmoid<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub unsafe fn new(
        input_ptr: *const T,
        gate_weight_ptr: *const T,
        bias_ptr: Option<*const T>,
        output_ptr: *mut T,
        params: MatMulParams,
        m_max: usize,
        n_max: usize,
        k_max: usize,
        use_routing_bias: bool,
        decode_only_flag: bool,
    ) -> Self {
        let reduction_block_cols = params.kc();
        let micro_tile_cols = params.nr();
        let input_block_rows = params.mb();
        let output_block_cols = params.nb();
        let b_panel_stride_elems = reduction_block_cols * micro_tile_cols;
        let acc_stride_elems = input_block_rows * output_block_cols;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let pool_len = threads * b_panel_stride_elems;
        let acc_pool_len = threads * acc_stride_elems;
        let b_panel_pool: Vec<T> = vec![T::default(); pool_len];
        let acc_pool: Vec<T> = vec![T::default(); acc_pool_len];

        Self {
            ptr1: ConstPtr { ptr: input_ptr },
            ptr2: ConstPtr {
                ptr: gate_weight_ptr,
            },
            output_ptr: MutPtr { ptr: output_ptr },
            params,
            m_max,
            n_max,
            k_max,
            _marker: PhantomData,
            b_panel_pool: b_panel_pool.into_boxed_slice(),
            b_panel_stride_elems,
            acc_pool: acc_pool.into_boxed_slice(),
            acc_stride_elems,
            bias_ptr: bias_ptr.map(|ptr| ConstPtr { ptr }),
            use_routing_bias,
            decode_only_flag,
        }
    }
}

impl<T> MatMulSigmoid<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
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
            let input_block_rows = self.params.mb();
            let output_block_cols = self.params.nb();
            let micro_tile_rows = self.params.mr();

            // Pad rows so the scalar block kernel sees aligned micro tiles.
            // 补齐行数，让 scalar block kernel 看到对齐的微内核 tile。
            let padded_input_rows = active_input_rows.div_ceil(micro_tile_rows) * micro_tile_rows;
            debug_assert!(padded_input_rows <= self.m_max);
            debug_assert!(input_block_rows % micro_tile_rows == 0);
            debug_assert!(output_cols % self.params.nr() == 0);
            debug_assert!(reduction_cols % self.params.kc() == 0);

            let max_threads = if self.b_panel_stride_elems == 0 {
                0
            } else {
                self.b_panel_pool.len() / self.b_panel_stride_elems
            };

            debug_assert!(thread_num >= 1);
            debug_assert!(thread_id < thread_num);
            debug_assert!(thread_num <= max_threads);

            let input_tile_count = padded_input_rows.div_ceil(input_block_rows);
            let output_tile_count = output_cols.div_ceil(output_block_cols);
            let total_tiles = input_tile_count * output_tile_count;

            let b_panel_ptr = self
                .b_panel_pool
                .as_ptr()
                .add(thread_id * self.b_panel_stride_elems) as *mut T;
            let acc_ptr = self
                .acc_pool
                .as_ptr()
                .add(thread_id * self.acc_stride_elems) as *mut T;
            let bias_ptr = self.bias_ptr.map(|ptr| ptr.ptr);

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

                    debug_assert!(input_rows_in_block % micro_tile_rows == 0);
                    debug_assert!(output_cols_in_block % self.params.nr() == 0);

                    kernel::scalar::block_matmul_sigmoid::matmul_sigmoid(
                        self.ptr1.ptr,
                        self.ptr2.ptr,
                        self.output_ptr.ptr,
                        &self.params,
                        self.m_max,
                        self.n_max,
                        self.k_max,
                        bias_ptr,
                        self.use_routing_bias,
                        input_row_start,
                        output_col_start,
                        input_rows_in_block,
                        output_cols_in_block,
                        b_panel_ptr,
                        acc_ptr,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_matmul_sigmoid_runner_f32_nt_bias() {
        const M: usize = 3;
        const K: usize = 64;
        const N: usize = 32;

        let mut a = vec![0.0f32; M * K];
        let mut b_nt = vec![0.0f32; N * K];
        let mut bias = vec![0.0f32; N];
        let mut c = vec![0.0f32; M * N];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = 0.01 * (i as f32) + 0.001 * (kk as f32);
            }
        }
        for j in 0..N {
            for kk in 0..K {
                b_nt[j * K + kk] = 0.02 * (kk as f32) + 0.003 * (j as f32);
            }
            bias[j] = 0.05 * (j as f32);
        }

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let runner = unsafe {
            MatMulSigmoid::<f32>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                Some(bias.as_ptr()),
                c.as_mut_ptr(),
                params,
                M,
                N,
                K,
                true,
                false,
            )
        };

        runner.run(M, 0, 1, 0);

        for i in 0..M {
            for j in 0..N {
                let mut sum = bias[j];
                for kk in 0..K {
                    sum += a[i * K + kk] * b_nt[j * K + kk];
                }
                let expected = 1.0f32 / (1.0f32 + (-sum).exp());
                let got = c[i * N + j];
                assert_abs_diff_eq!(got, expected, epsilon = 1e-5);
            }
        }
    }
}
