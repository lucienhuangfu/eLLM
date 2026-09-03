use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::kernel::common::matmul_params::MatMulParams;
use crate::mem_mgr::allocator::AlignedBox;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::Sigmoid;
use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};

/// Gate projection with sigmoid activation (router scoring).
/// Pre-packs the gate weight into MemPool (replacing the original),
/// so the operator only holds a pointer — no duplicate copy.
/// Gate 投影 + sigmoid 激活。权重预打包存入 MemPool（替换原始），
/// operator 只持指针 — 无双份拷贝。
#[derive(Clone)]
pub struct MatMulSigmoid<T> {
    pub ptr1: ConstPtr<T>,     // Input matrix: [input_rows, reduction_cols].
    pub output_ptr: MutPtr<T>, // Sigmoid output: [input_rows, output_cols].
    pub params: MatMulParams,
    pub m_max: usize,
    pub n_max: usize,
    pub k_max: usize,
    pub _marker: PhantomData<T>,

    // Pointer to packed gate weight in MemPool (no separate Box<[T]>).
    // 指向 MemPool 中已打包 gate 权重的指针（无独立 Box<[T]>）。
    packed_ptr: ConstPtr<T>,
    packed_panel_stride: usize,

    bias_ptr: Option<ConstPtr<T>>,
    use_routing_bias: bool,
    decode_only_flag: bool,
}

impl<T> MatMulSigmoid<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + GlobalMemPool,
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
        weight_name: String,
    ) -> Self {
        let reduction_block_cols = params.kc().max(1);
        let micro_tile_cols = params.nr().max(1);
        let packed_panel_stride = reduction_block_cols * micro_tile_cols;

        // Pack into AlignedBox, then store in MemPool (replacing original).
        // Pack 到 AlignedBox，然后存入 MemPool 替换原始权重。
        let packed_ptr = Self::pack_into_pool(
            gate_weight_ptr,
            n_max,
            k_max,
            reduction_block_cols,
            micro_tile_cols,
            &weight_name,
        );

        Self {
            ptr1: ConstPtr { ptr: input_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            params,
            m_max,
            n_max,
            k_max,
            _marker: PhantomData,
            packed_ptr: ConstPtr { ptr: packed_ptr },
            packed_panel_stride,
            bias_ptr: bias_ptr.map(|ptr| ConstPtr { ptr }),
            use_routing_bias,
            decode_only_flag,
        }
    }

    /// Pack gate weight into an AlignedBox and store in MemPool,
    /// replacing (and freeing) the original. Returns (pointer, AlignedBox).
    /// 将 gate 权重打包到 AlignedBox 并存入 MemPool，
    /// 替换（并释放）原始数据。
    fn pack_into_pool(
        weight_nt: *const T,
        output_cols: usize,
        reduction_cols: usize,
        reduction_block_cols: usize,
        micro_tile_cols: usize,
        weight_name: &str,
    ) -> *const T {
        let reduction_panel_count = reduction_cols.div_ceil(reduction_block_cols);
        let output_panel_count = output_cols.div_ceil(micro_tile_cols);
        let panel_stride = reduction_block_cols * micro_tile_cols;
        let total_size = reduction_panel_count * output_panel_count * panel_stride;
        let mut packed = AlignedBox::<T>::allocate_zero(total_size);

        unsafe {
            let dst = packed.as_mut_ptr();
            for reduction_panel_index in 0..reduction_panel_count {
                let reduction_start = reduction_panel_index * reduction_block_cols;
                let reduction_cols_this =
                    (reduction_cols - reduction_start).min(reduction_block_cols);
                for output_panel_index in 0..output_panel_count {
                    let output_start = output_panel_index * micro_tile_cols;
                    let output_cols_this = (output_cols - output_start).min(micro_tile_cols);
                    let panel = dst.add(
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

        let ptr = packed.as_ptr();
        // Replace original weight with packed version in MemPool.
        // 用打包版本替换 MemPool 中的原始权重。
        T::with_global(|pool| {
            pool.remove(weight_name);
            pool.replace_weight(weight_name, packed);
        });
        ptr
    }

}

impl<T> MatMulSigmoid<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    #[inline(always)]
    fn packed_panel_ptr(&self, output_col_start: usize, reduction_col_start: usize) -> *const T {
        let reduction_block_cols = self.params.kc().max(1);
        let micro_tile_cols = self.params.nr().max(1);
        let output_panel_count = self.n_max.div_ceil(micro_tile_cols);
        let panel_index = (reduction_col_start / reduction_block_cols) * output_panel_count
            + (output_col_start / micro_tile_cols);
        unsafe {
            self.packed_ptr
                .ptr
                .add(panel_index * self.packed_panel_stride)
        }
    }
}

impl<T> MatMulSigmoid<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sigmoid,
{
    /// Run the pre-packed sigmoid-gate projection.
    /// Uses pre-packed weight panels — no on-the-fly packing.
    /// 运行预打包的 sigmoid-gate 投影。使用预打包权重面板，
    /// 无需运行时打包。
    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let active_input_rows = if prefill_size == 0 {
                decode_size
            } else {
                prefill_size
            };

            let output_cols = self.n_max;
            let reduction_cols = self.k_max;
            let input_block_rows = self.params.mb();
            let output_block_cols = self.params.nb();
            let reduction_block_cols = self.params.kc().max(1);
            let micro_tile_rows = self.params.mr().max(1);
            let micro_tile_cols = self.params.nr().max(1);

            let padded_input_rows = active_input_rows.div_ceil(micro_tile_rows) * micro_tile_rows;
            debug_assert!(padded_input_rows <= self.m_max);
            debug_assert!(input_block_rows % micro_tile_rows == 0);
            debug_assert!(output_cols % micro_tile_cols == 0);
            debug_assert!(reduction_cols % reduction_block_cols == 0);

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

                    // Zero-init output tile (accumulator).
                    // 输出 tile 零初始化（累加器）。
                    let mut output_col_offset = 0;
                    while output_col_offset < output_cols_in_block {
                        let mut input_row_offset = 0;
                        while input_row_offset < input_rows_in_block {
                            let output_tile = output_base.add(
                                (input_row_start + input_row_offset) * output_row_stride
                                    + (output_col_start + output_col_offset),
                            );
                            for row_in_tile in 0..micro_tile_rows {
                                let output_row = output_tile.add(row_in_tile * output_row_stride);
                                for col_in_tile in 0..micro_tile_cols {
                                    *output_row.add(col_in_tile) = T::default();
                                }
                            }
                            input_row_offset += micro_tile_rows;
                        }
                        output_col_offset += micro_tile_cols;
                    }

                    // Accumulate over reduction panels using pre-packed weights.
                    // 使用预打包权重在 reduction panel 上累加。
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

                                // MR × NR micro-kernel accumulate.
                                for row_in_tile in 0..micro_tile_rows {
                                    let a_row = input_tile.add(row_in_tile * input_row_stride);
                                    let c_row =
                                        output_tile.add(row_in_tile * output_row_stride);
                                    for col_in_tile in 0..micro_tile_cols {
                                        let mut sum = *c_row.add(col_in_tile);
                                        let b_row = weight_panel_ptr
                                            .add(col_in_tile * reduction_block_cols);
                                        // Actually the panel layout is [KC][NR], so:
                                        // b[k * NR + col] — reduction_lane × NR + output_lane
                                        // Wait, check: packed_panel layout is reduction-major,
                                        // each reduction_lane has NR output_lanes.
                                        // The micro-kernel reads b_row = panel + col_lane * KC.
                                        // No, let me fix this.
                                        for k in 0..reduction_block_cols {
                                            sum = sum
                                                + *a_row.add(k)
                                                    * *weight_panel_ptr.add(
                                                        k * micro_tile_cols + col_in_tile,
                                                    );
                                        }
                                        *c_row.add(col_in_tile) = sum;
                                    }
                                }

                                input_row_offset += micro_tile_rows;
                            }
                            output_col_offset += micro_tile_cols;
                        }
                        reduction_col_start += reduction_block_cols;
                    }

                    // Apply sigmoid (and optional bias) to the accumulated tile.
                    // 对累加后的 tile 应用 sigmoid（和可选的 bias）。
                    let mut output_col_offset = 0;
                    while output_col_offset < output_cols_in_block {
                        let mut input_row_offset = 0;
                        while input_row_offset < input_rows_in_block {
                            let output_tile = output_base.add(
                                (input_row_start + input_row_offset) * output_row_stride
                                    + (output_col_start + output_col_offset),
                            );
                            for row_in_tile in 0..micro_tile_rows {
                                let c_row =
                                    output_tile.add(row_in_tile * output_row_stride);
                                if self.use_routing_bias {
                                    if let Some(bias_ptr) = self.bias_ptr {
                                        let bias_row = bias_ptr.ptr.add(output_col_start);
                                        for col_in_tile in 0..micro_tile_cols {
                                            let col = output_col_start + output_col_offset + col_in_tile;
                                            if col < output_cols {
                                                let val = *c_row.add(col_in_tile)
                                                    + *bias_row.add(col_in_tile);
                                                *c_row.add(col_in_tile) = val.sigmoid();
                                            }
                                        }
                                    }
                                } else {
                                    for col_in_tile in 0..micro_tile_cols {
                                        let col = output_col_start + output_col_offset + col_in_tile;
                                        if col < output_cols {
                                            *c_row.add(col_in_tile) =
                                                (*c_row.add(col_in_tile)).sigmoid();
                                        }
                                    }
                                }
                            }
                            input_row_offset += micro_tile_rows;
                        }
                        output_col_offset += micro_tile_cols;
                    }
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
        use std::collections::HashMap;
        f32::init_global(HashMap::new());

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

        // Pre-load weight into MemPool so pack_into_pool can replace it.
        let _weight_tensor = crate::tensor::Tensor::<f32>::from_vec(
            vec![N, K],
            b_nt.clone(),
            "test.sigmoid_gate.weight".to_string(),
        );

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
                "test.sigmoid_gate.weight".to_string(),
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
