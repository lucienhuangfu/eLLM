use std::marker::PhantomData;
use std::ops::{Add, Div, Mul};

use crate::num_traits::Exp;
use crate::operators::assign::{assign, assign_slice_channel_tile};
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::RecurrentGatedDeltaRuleTrait;
use crate::runtime::SequenceSlice;

// Fused single-step gated delta rule recurrence for GatedDeltaNet-style
// linear attention layers: the decode path of the reference
// torch_recurrent_gated_delta_rule (scripts/modeling_qwen3_5_moe.py, the
// seq_len == 1 branch). For every token row and every v head h it runs:
// 面向 GatedDeltaNet 类线性注意力层的融合单步 gated delta rule 递推：
// 对应参考实现 torch_recurrent_gated_delta_rule 的 decode 路径
// （scripts/modeling_qwen3_5_moe.py 中 seq_len == 1 分支）。
// 对每个 token 行、每个 v 头 h 执行：
//
//   S_h <- exp(g_h) * S_h            gating decay of the recurrent state
//   e   <- v - S_h^T * k             delta error             [head_v_dim]
//   S_h <- S_h + beta_h * k ⊗ e      delta-rule state update
//   o_h <- S_h^T * q                 attention output        [head_v_dim]
//   o_h <- w ⊙ o_h * rsqrt(mean(o_h^2) + eps) * silu(z_h)
//                                    fused gated RMSNorm epilogue
//
// The final step absorbs the standalone gated RMSNorm (reference
// Qwen3_5MoeRMSNormGated: norm per head over head_v_dim, gate from the
// z branch, applied right before out_proj) as an output epilogue of
// this operator; the norm weight and the z branch are therefore inputs
// of this operator rather than a separate RMSGatedZipMap.
// 最后一步把独立的 gated RMSNorm（参考实现 Qwen3_5MoeRMSNormGated：
// 沿 head_v_dim 按头归一化、z 支路作门控，紧接在 out_proj 之前）
// 吸收为本算子的输出 epilogue；归一化权重与 z 支路因此是本算子的输入，
// 不再有单独的 RMSGatedZipMap 算子。
//
// The recurrent state is the layer's cache and is updated in place,
// doubling as initial_state / output_final_state of the reference call:
// state[batch_size, num_v_heads, head_k_dim, head_v_dim].
// 递推状态即该层的缓存，原地更新，兼作参考调用中的
// initial_state / output_final_state：
// state[batch_size, num_v_heads, head_k_dim, head_v_dim]。
//
// Inputs come straight from the upstream operators: the qkv cache rows
// were convolved + SiLU'd and already carry the l2-norm (+ query scale)
// epilogue of CausalConv1dSilu, so this operator must NOT normalize
// again (matching use_qk_l2norm_in_kernel handled upstream); g and beta
// are per-token per-v-head scalars with the same cache row layout.
// 输入直接来自上游算子：qkv 缓存行已经过卷积 + SiLU，并已带
// CausalConv1dSilu 的 l2 归一化（+ query 缩放）epilogue，
// 因此本算子不再重复归一化（对应 use_qk_l2norm_in_kernel 在上游处理）；
// g 与 beta 是逐 token 逐 v 头的标量，与缓存行布局一致。
//
// Rows of one sequence share the state of their batch and must run in
// order, so the rows inside a slice run sequentially; v heads are
// independent (each owns its state and one q/k head via the GQA mapping
// kv_head = h * num_k_heads / num_v_heads, mirroring repeat_interleave)
// and become the second parallel dimension when slices alone cannot
// fill the thread pool.
// 同一序列的行共享所属 batch 的状态，必须按序执行，因此 slice 内部
// 的行串行处理；v 头之间相互独立（每个头拥有自己的状态，并通过
// GQA 映射 kv_head = h * num_k_heads / num_v_heads 复用一个 q/k 头，
// 对应 repeat_interleave），仅靠 slice 填不满线程池时，v 头成为
// 第二个并行维度。

#[derive(Clone)]
pub struct RecurrentGatedDeltaRule<T> {
    pub qkv_ptr: ConstPtr<T>, // qkv cache after CausalConv1dSilu: [sequence_length * batch_size, conv_dim].
    pub g_ptr: ConstPtr<T>,   // log gating decay: [sequence_length * batch_size, num_v_heads].
    pub beta_ptr: ConstPtr<T>, // delta-rule learning rate: [sequence_length * batch_size, num_v_heads].
    pub state_ptr: MutPtr<T>, // [batch_size, num_v_heads, head_k_dim, head_v_dim], updated in place.
    pub output_ptr: MutPtr<T>, // [sequence_length * batch_size, value_dim].

    // Split of the qkv row: q occupies [0, key_dim), k occupies
    // [key_dim, 2 * key_dim), v occupies [2 * key_dim, conv_dim).
    // qkv 行的分段：q 占 [0, key_dim)、k 占 [key_dim, 2 * key_dim)、
    // v 占 [2 * key_dim, conv_dim)。
    pub key_dim: usize,   // num_k_heads * head_k_dim
    pub value_dim: usize, // num_v_heads * head_v_dim

    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,

    // Shape of the qkv cache, matching MatMulProj / CausalConv1dSilu.
    // qkv 缓存的形状，与 MatMulProj / CausalConv1dSilu 保持一致。
    pub sequence_length: usize,
    pub batch_size: usize,
    pub _marker: PhantomData<T>,
}

impl<T> RecurrentGatedDeltaRule<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Exp,
{
    pub unsafe fn new(
        qkv_ptr: *const T,  // [sequence_length * batch_size, conv_dim]
        g_ptr: *const T,    // [sequence_length * batch_size, num_v_heads]
        beta_ptr: *const T, // [sequence_length * batch_size, num_v_heads]
        state_ptr: *mut T,  // [batch_size, num_v_heads, head_k_dim, head_v_dim]
        output_ptr: *mut T, // [sequence_length * batch_size, value_dim]
        key_dim: usize,
        value_dim: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        sequence_length: usize,
        batch_size: usize,
    ) -> Self {
        debug_assert!(head_k_dim > 0);
        debug_assert!(head_v_dim > 0);
        debug_assert!(num_k_heads > 0);
        debug_assert_eq!(num_v_heads % num_k_heads, 0);
        debug_assert_eq!(key_dim, num_k_heads * head_k_dim);
        debug_assert_eq!(value_dim, num_v_heads * head_v_dim);

        Self {
            qkv_ptr: ConstPtr { ptr: qkv_ptr },
            g_ptr: ConstPtr { ptr: g_ptr },
            beta_ptr: ConstPtr { ptr: beta_ptr },
            state_ptr: MutPtr { ptr: state_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            key_dim,
            value_dim,
            head_k_dim,
            head_v_dim,
            num_k_heads,
            num_v_heads,
            sequence_length,
            batch_size,
            _marker: PhantomData,
        }
    }

    // Row width of the qkv cache: q + k + v segments.
    // qkv 缓存的行宽：q + k + v 三段。
    #[inline(always)]
    fn conv_dim(&self) -> usize {
        2 * self.key_dim + self.value_dim
    }

    pub fn run(
        &self,
        _total_size: usize,
        attention_list: &[SequenceSlice],
        thread_num: usize,
        thread_id: usize,
    ) {
        debug_assert!(thread_num >= 1);
        debug_assert!(thread_id < thread_num);

        if attention_list.is_empty() {
            return;
        }

        // Enough slices to fill the pool: keep the slice as the unit, one
        // thread per slice region, full head range per thread.
        // slice 足以填满线程池：保持 slice 为调度单位，每线程处理完整头范围。
        if attention_list.len() >= thread_num {
            if let Some((slice_begin, slice_end)) =
                assign(attention_list.len(), thread_num, thread_id)
            {
                for slice in &attention_list[slice_begin..slice_end] {
                    self.run_slice(slice, 0, self.num_v_heads);
                }
            }
            return;
        }

        // Slices alone cannot fill the pool (e.g. small-batch decode):
        // distribute threads across slices proportionally to their row
        // count, then split the v-head dimension inside each slice. Rows
        // inside a slice stay sequential per head block, so blocks never
        // race on the recurrent state.
        // 仅靠 slice 填不满线程池（如小 batch decode）：按行数比例给各
        // slice 分线程，再在 slice 内部切分 v 头维度。每个头块内的行
        // 保持串行，块之间不会竞争递推状态。
        let slice_lengths: Vec<usize> = attention_list.iter().map(|s| s.length).collect();
        let max_blocks: Vec<usize> = slice_lengths.iter().map(|_| self.num_v_heads).collect();
        if let Some(tile) =
            assign_slice_channel_tile(&slice_lengths, &max_blocks, thread_num, thread_id)
        {
            if let Some((head_begin, head_end)) =
                assign(self.num_v_heads, tile.local_num, tile.local_id)
            {
                self.run_slice(&attention_list[tile.slice_index], head_begin, head_end);
            }
        }
    }

    // Processes the rows of one slice sequentially, restricted to the
    // v-head range [head_begin, head_end).
    // 按序处理单个 slice 的所有行，只覆盖 [head_begin, head_end) 的 v 头。
    fn run_slice(&self, slice: &SequenceSlice, head_begin: usize, head_end: usize) {
        for offset in 0..slice.length {
            let next_sequence_index = slice.next_sequence_index + offset;
            if next_sequence_index >= self.sequence_length {
                continue;
            }

            unsafe {
                // Same cache row placement as MatMulProj's qkv output.
                // 与 MatMulProj 的 qkv 输出采用相同的缓存行落位。
                let cache_row = next_sequence_index * self.batch_size + slice.batch_index;
                let qkv_row_ptr = self.qkv_ptr.ptr.add(cache_row * self.conv_dim());
                let g_row_ptr = self.g_ptr.ptr.add(cache_row * self.num_v_heads);
                let beta_row_ptr = self.beta_ptr.ptr.add(cache_row * self.num_v_heads);
                let state_ptr = self
                    .state_ptr
                    .ptr
                    .add(slice.batch_index * self.num_v_heads * self.head_k_dim * self.head_v_dim);
                let output_row_ptr = self.output_ptr.ptr.add(cache_row * self.value_dim);
                self.compute(
                    qkv_row_ptr,
                    g_row_ptr,
                    beta_row_ptr,
                    state_ptr,
                    output_row_ptr,
                    head_begin,
                    head_end,
                );
            }
        }
    }
}

impl<T> RecurrentGatedDeltaRuleTrait<T> for RecurrentGatedDeltaRule<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Exp,
{
    default fn compute(
        &self,
        _qkv_row_ptr: *const T,
        _g_row_ptr: *const T,
        _beta_row_ptr: *const T,
        _state_ptr: *mut T,
        _output_ptr: *mut T,
        _head_begin: usize,
        _head_end: usize,
    ) {
        // TODO: compute logic, filled in later
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recurrent_gated_delta_rule_construct_and_partition() {
        const HEAD_K_DIM: usize = 2;
        const HEAD_V_DIM: usize = 2;
        const NUM_K_HEADS: usize = 2;
        const NUM_V_HEADS: usize = 4; // num_v_heads / num_k_heads = 2, exercises the GQA mapping
        const KEY_DIM: usize = NUM_K_HEADS * HEAD_K_DIM;
        const VALUE_DIM: usize = NUM_V_HEADS * HEAD_V_DIM;
        const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;
        const SEQUENCE_LENGTH: usize = 4;
        const BATCH_SIZE: usize = 1;

        let qkv_data: Vec<f32> = (0..SEQUENCE_LENGTH * BATCH_SIZE * CONV_DIM)
            .map(|x| (x % 7) as f32 * 0.01)
            .collect();
        let g_data = vec![0.0f32; SEQUENCE_LENGTH * BATCH_SIZE * NUM_V_HEADS];
        let beta_data = vec![1.0f32; SEQUENCE_LENGTH * BATCH_SIZE * NUM_V_HEADS];
        let mut state_data = vec![0.0f32; BATCH_SIZE * NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM];
        let mut output_data = vec![0.0f32; SEQUENCE_LENGTH * BATCH_SIZE * VALUE_DIM];

        let operator = unsafe {
            RecurrentGatedDeltaRule::<f32>::new(
                qkv_data.as_ptr(),
                g_data.as_ptr(),
                beta_data.as_ptr(),
                state_data.as_mut_ptr(),
                output_data.as_mut_ptr(),
                KEY_DIM,
                VALUE_DIM,
                HEAD_K_DIM,
                HEAD_V_DIM,
                NUM_K_HEADS,
                NUM_V_HEADS,
                SEQUENCE_LENGTH,
                BATCH_SIZE,
            )
        };
        assert_eq!(operator.conv_dim(), CONV_DIM);

        // compute is still empty: running all threads over the slice must
        // not panic and must leave state and output untouched.
        // compute 仍为空：所有线程跑完该 slice 不应 panic，
        // 状态与输出保持原样。
        let thread_num = 4;
        let attention_list = [SequenceSlice {
            token_start_index: 0,
            batch_index: 0,
            next_sequence_index: 0,
            length: SEQUENCE_LENGTH,
            last_token_flag: true,
            lift_index: 0,
        }];
        for thread_id in 0..thread_num {
            operator.run(SEQUENCE_LENGTH, &attention_list, thread_num, thread_id);
        }
        assert!(state_data.iter().all(|&value| value == 0.0));
        assert!(output_data.iter().all(|&value| value == 0.0));
    }
}
