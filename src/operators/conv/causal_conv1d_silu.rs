use std::marker::PhantomData;
use std::ops::{Add, Mul};

use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::CausalConvTrait;

// Fused depthwise causal conv1d + SiLU + rolling state update for
// GatedDeltaNet-style linear attention layers (the mixed_qkv branch).
// One kernel replaces the separate conv1d, activation and conv_state
// bookkeeping: per channel it reads the rolling window of the previous
// kernel_size - 1 tokens, convolves with the channel weight, applies SiLU,
// and shifts the new token into the window.
// 面向 GatedDeltaNet 类线性注意力层（mixed_qkv 支路）的融合
// depthwise 因果卷积 + SiLU + 滚动状态更新。
// 单 kernel 取代独立的 conv1d、激活与 conv_state 维护：逐通道读取前
// kernel_size - 1 个 token 的滚动窗口，与通道权重卷积，应用 SiLU，
// 并把新 token 移入窗口。
//
// Token rows are processed against a per-sequence rolling window
// (kernel_size = 4 in Qwen3_5 MoE, state holds the previous 3 tokens).
// token 行针对所属序列的滚动窗口计算
// （Qwen3_5 MoE 中 kernel_size = 4，状态保存前 3 个 token）。

#[derive(Clone)]
pub struct CausalConv1dSilu<T> {
    pub input_ptr: ConstPtr<T>,  // Input rows: [token_rows, conv_dim].
    pub weight_ptr: ConstPtr<T>, // Depthwise weight: [conv_dim, kernel_size].
    pub state_ptr: MutPtr<T>,    // Rolling window: [conv_dim, kernel_size - 1].
    pub output_ptr: MutPtr<T>,   // Output rows: [token_rows, conv_dim].

    pub kernel_size: usize,
    pub conv_dim: usize,
    pub m_max: usize, // Maximum token rows per round. 每轮最大 token 行数。
    pub _marker: PhantomData<T>,
}

impl<T> CausalConv1dSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub unsafe fn new(
        input_ptr: *const T,  // [token_rows, conv_dim]
        weight_ptr: *const T, // [conv_dim, kernel_size]
        state_ptr: *mut T,    // [conv_dim, kernel_size - 1]
        output_ptr: *mut T,   // [token_rows, conv_dim]
        kernel_size: usize,
        conv_dim: usize,
        m_max: usize,
    ) -> Self {
        debug_assert!(kernel_size >= 2);

        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            weight_ptr: ConstPtr { ptr: weight_ptr },
            state_ptr: MutPtr { ptr: state_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            kernel_size,
            conv_dim,
            m_max,
            _marker: PhantomData,
        }
    }

    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        _total_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        let active_rows = if prefill_size == 0 {
            decode_size
        } else {
            prefill_size
        };

        debug_assert!(active_rows <= self.m_max);
        debug_assert!(thread_num >= 1);
        debug_assert!(thread_id < thread_num);

        if let Some((row_begin, row_end)) = assign(active_rows, thread_num, thread_id) {
            for row in row_begin..row_end {
                unsafe {
                    let input_row_ptr = self.input_ptr.ptr.add(row * self.conv_dim);
                    let output_row_ptr = self.output_ptr.ptr.add(row * self.conv_dim);
                    // TODO: multi-sequence batches need the per-sequence state offset
                    // from computing_slices; the skeleton targets a single sequence.
                    // TODO: 多序列 batch 需要从 computing_slices 取各序列的状态偏移；
                    // 骨架阶段面向单序列。
                    let state_ptr = self.state_ptr.ptr;
                    self.compute(
                        input_row_ptr,
                        self.weight_ptr.ptr,
                        state_ptr,
                        output_row_ptr,
                    );
                }
            }
        }
    }
}

impl<T> CausalConvTrait<T> for CausalConv1dSilu<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(
        &self,
        _input_ptr: *const T,
        _weight_ptr: *const T,
        _state_ptr: *mut T,
        _output_ptr: *mut T,
    ) {
        // TODO: compute logic, filled in later
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_conv1d_silu_construct_and_partition() {
        const M: usize = 4;
        const CONV_DIM: usize = 6;
        const KERNEL_SIZE: usize = 4;

        let input_data: Vec<f32> = (0..M * CONV_DIM).map(|x| (x % 7) as f32 * 0.01).collect();
        let weight_data: Vec<f32> = (0..CONV_DIM * KERNEL_SIZE)
            .map(|x| (x % 5) as f32 * 0.01)
            .collect();
        let mut state_data = vec![0.0f32; CONV_DIM * (KERNEL_SIZE - 1)];
        let mut output_data = vec![0.0f32; M * CONV_DIM];

        let operator = unsafe {
            CausalConv1dSilu::<f32>::new(
                input_data.as_ptr(),
                weight_data.as_ptr(),
                state_data.as_mut_ptr(),
                output_data.as_mut_ptr(),
                KERNEL_SIZE,
                CONV_DIM,
                M,
            )
        };
        assert_eq!(operator.kernel_size, KERNEL_SIZE);

        // While compute is empty: only assert no panic / partition coverage.
        // compute 为空期间：只验证不 panic、分区覆盖完整。
        let thread_num = 4;
        for thread_id in 0..thread_num {
            operator.run(M, 0, M, thread_num, thread_id);
        }
        assert!(output_data.iter().all(|&value| value == 0.0));
        assert!(state_data.iter().all(|&value| value == 0.0));
    }
}
