use std::marker::PhantomData;
use std::ops::{Add, Div, Mul};

use crate::num_traits::{FromNumber, Sqrt};
use crate::operators::assign::{assign, assign_slice_channel_tile};
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::CausalConvTrait;
use crate::runtime::SequenceSlice;

// Minimum channels kept inside one scheduling block; blocks thinner than
// this would waste the SIMD width of the inner channel loop.
// 单个调度块至少包含的通道数；比这更碎的块会浪费内层通道循环的 SIMD 宽度。
const MIN_CHANNEL_BLOCK: usize = 64;

// Fused depthwise causal conv1d + SiLU + rolling state update for
// GatedDeltaNet-style linear attention layers (the mixed_qkv branch).
// It consumes the qkv cache produced by the upstream MatMulProj: the same
// buffer is read and written in place, one cache row per
// (sequence_position, batch) pair, indexed as
// qkv[(next_sequence_index * batch_size + batch_index) * conv_dim],
// exactly the layout MatMulProj writes (same as the KV cache in MatMul3).
// 面向 GatedDeltaNet 类线性注意力层（mixed_qkv 支路）的融合
// depthwise 因果卷积 + SiLU + 滚动状态更新。
// 直接消费上游 MatMulProj 写入的 qkv 缓存：同一缓冲原地读写，每个
// (序列位置, batch) 一行，寻址为
// qkv[(next_sequence_index * batch_size + batch_index) * conv_dim]，
// 与 MatMulProj 的落位布局完全一致（同 MatMul3 的 KV cache）。
//
// One kernel replaces the separate conv1d, activation and conv_state
// bookkeeping: per channel it reads the rolling window of the previous
// kernel_size - 1 tokens, convolves with the channel weight, applies SiLU,
// and shifts the new token into the window. Rows of one sequence share the
// rolling state of their batch and must stay in order, so the rows inside
// a slice run sequentially; channels are independent and become the second
// parallel dimension when slices alone cannot fill the thread pool.
// 单 kernel 取代独立的 conv1d、激活与 conv_state 维护：逐通道读取前
// kernel_size - 1 个 token 的滚动窗口，与通道权重卷积，应用 SiLU，
// 并把新 token 移入窗口。同一序列的行共享所属 batch 的滚动状态，必须
// 按序执行，因此 slice 内部的行串行处理；仅靠 slice 填不满线程池时，通道成为第二个并行维度。
//
// The rolling window is kept per batch:
// state[batch_size, conv_dim, kernel_size - 1]
// (kernel_size = 4 in Qwen3_5 MoE, state holds the previous 3 tokens).
// 滚动窗口按 batch 独立保存：
// state[batch_size, conv_dim, kernel_size - 1]
// （Qwen3_5 MoE 中 kernel_size = 4，状态保存前 3 个 token）。
//
// Right after conv + SiLU the kernel also applies the qk-norm epilogue the
// reference runs just after the split: every head of the q segment
// [0, key_dim) and the k segment [key_dim, 2 * key_dim) is l2-normalized
// in place along its head_k_dim channels, and the q heads additionally
// take the attention scale 1 / sqrt(head_k_dim):
//   q_h *= (1 / sqrt(head_k_dim)) * rsqrt(sum(q_h^2) + eps)
//   k_h *= rsqrt(sum(k_h^2) + eps)                 (eps = 1e-6)
// The value segment is untouched. A norm needs the whole head, so the
// channel tiling is aligned to head boundaries: every scheduling block
// spans a whole number of heads and each head is normalized by exactly
// one thread (conv_dim must be a multiple of head_k_dim).
// 卷积 + SiLU 之后，同一 kernel 内紧跟参考实现在 split 之后做的 qk 归一化
// epilogue：q 段 [0, key_dim) 与 k 段 [key_dim, 2 * key_dim) 的每个头沿其
// head_k_dim 个通道原地 l2 归一化，q 头额外乘注意力缩放 1 / sqrt(head_k_dim)：
//   q_h *= (1 / sqrt(head_k_dim)) * rsqrt(sum(q_h^2) + eps)
//   k_h *= rsqrt(sum(k_h^2) + eps)                 (eps = 1e-6)
// value 段不动。归一化需要完整的一个头，因此通道分块按头边界对齐：
// 每个调度块覆盖整数个头，每个头恰好由一个线程归一化
// （conv_dim 必须是 head_k_dim 的整数倍）。

#[derive(Clone)]
pub struct CausalConv1dSilu<T> {
    pub qkv_ptr: MutPtr<T>, // qkv cache from MatMulProj, read/written in place.
    pub weight_ptr: ConstPtr<T>, // Depthwise weight: [conv_dim, kernel_size].
    pub state_ptr: MutPtr<T>, // [batch_size, conv_dim, kernel_size - 1].

    pub kernel_size: usize,
    pub conv_dim: usize, // == MatMulProj::qkv_cols.

    // Split point and per-head width of the q/k segments: q occupies
    // [0, key_dim), k occupies [key_dim, 2 * key_dim), both laid out as
    // consecutive heads of head_k_dim channels.
    // q/k 段的分割点与每头宽度：q 占 [0, key_dim)、k 占 [key_dim, 2 * key_dim)，
    // 均为 head_k_dim 通道的连续头。
    pub key_dim: usize,
    pub head_k_dim: usize,

    // Shape of the qkv cache, matching MatMulProj.
    // qkv 缓存的形状，与 MatMulProj 保持一致。
    pub sequence_length: usize,
    pub batch_size: usize,
    pub _marker: PhantomData<T>,
}

impl<T> CausalConv1dSilu<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Sqrt,
{
    pub unsafe fn new(
        qkv_ptr: *mut T,      // [sequence_length * batch_size, conv_dim]
        weight_ptr: *const T, // [conv_dim, kernel_size]
        state_ptr: *mut T,    // [batch_size, conv_dim, kernel_size - 1]
        kernel_size: usize,
        conv_dim: usize,
        key_dim: usize,
        head_k_dim: usize,
        sequence_length: usize,
        batch_size: usize,
    ) -> Self {
        debug_assert!(kernel_size >= 2);
        debug_assert!(head_k_dim > 0);
        debug_assert_eq!(key_dim % head_k_dim, 0);
        debug_assert_eq!(conv_dim % head_k_dim, 0);
        debug_assert!(conv_dim >= 2 * key_dim);

        Self {
            qkv_ptr: MutPtr { ptr: qkv_ptr },
            weight_ptr: ConstPtr { ptr: weight_ptr },
            state_ptr: MutPtr { ptr: state_ptr },
            kernel_size,
            conv_dim,
            key_dim,
            head_k_dim,
            sequence_length,
            batch_size,
            _marker: PhantomData,
        }
    }

    // Scheduling block width aligned to head boundaries: the smallest
    // multiple of head_k_dim that is at least MIN_CHANNEL_BLOCK, so blocks
    // stay SIMD friendly without cutting any head.
    // 按头边界对齐的调度块宽度：不小于 MIN_CHANNEL_BLOCK 的最小 head_k_dim
    // 倍数，块既对 SIMD 友好又不切断任何头。
    #[inline(always)]
    fn head_block_size(&self) -> usize {
        let heads_per_block = (MIN_CHANNEL_BLOCK + self.head_k_dim - 1) / self.head_k_dim;
        heads_per_block.max(1) * self.head_k_dim
    }

    // Number of head-aligned blocks covering conv_dim; caps the per-slice
    // thread count in run.
    // 覆盖 conv_dim 的头对齐块数；作为 run 中单 slice 的线程数上限。
    #[inline(always)]
    fn head_block_num(&self) -> usize {
        (self.conv_dim + self.head_block_size() - 1) / self.head_block_size()
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
        // thread per slice region, full channel range per thread.
        // slice 足以填满线程池：保持 slice 为调度单位，每线程处理完整通道。
        if attention_list.len() >= thread_num {
            if let Some((slice_begin, slice_end)) =
                assign(attention_list.len(), thread_num, thread_id)
            {
                for slice in &attention_list[slice_begin..slice_end] {
                    self.run_slice(slice, 0, self.conv_dim);
                }
            }
            return;
        }

        // Slices alone cannot fill the pool (e.g. small-batch decode or one
        // long prefill): distribute threads across slices proportionally to
        // their row count, then split the channel dimension inside each
        // slice. Rows inside a slice stay sequential per channel block, so
        // blocks never race on the rolling state.
        // 仅靠 slice 填不满线程池（如小 batch decode 或单条长 prefill）：按行数比例给各 slice 分线程，
        // 再在 slice 内部切分通道维度。每个通道块内的行保持串行，块之间不会竞争滚动状态。
        let slice_lengths: Vec<usize> = attention_list.iter().map(|s| s.length).collect();
        // Per-slice thread cap by head-aligned block granularity: blocks
        // thinner than MIN_CHANNEL_BLOCK would waste SIMD width, and blocks
        // cutting a head would break the norm epilogue.
        // 单 slice 线程数上限取头对齐的块粒度：比 MIN_CHANNEL_BLOCK 更碎的块会浪费 SIMD 宽度，
        // 而切断头的块会破坏归一化 epilogue。
        let max_blocks: Vec<usize> = slice_lengths
            .iter()
            .map(|_| self.head_block_num())
            .collect();
        if let Some(tile) =
            assign_slice_channel_tile(&slice_lengths, &max_blocks, thread_num, thread_id)
        {
            // Split the head dimension instead of raw channels so every
            // block boundary lands on a head edge, as the norm epilogue
            // needs whole heads.
            // 切分头维度而非裸通道，让每个块边界都落在头边界上，
            // 归一化 epilogue 需要完整的一个头。
            let head_num = self.conv_dim / self.head_k_dim;
            if let Some((head_begin, head_end)) = assign(head_num, tile.local_num, tile.local_id) {
                self.run_slice(
                    &attention_list[tile.slice_index],
                    head_begin * self.head_k_dim,
                    head_end * self.head_k_dim,
                );
            }
        }
    }

    // Processes the rows of one slice sequentially, restricted to the
    // channel range [channel_begin, channel_end).
    // 按序处理单个 slice 的所有行，只覆盖 [channel_begin, channel_end) 通道。
    fn run_slice(&self, slice: &SequenceSlice, channel_begin: usize, channel_end: usize) {
        for offset in 0..slice.length {
            let next_sequence_index = slice.next_sequence_index + offset;
            if next_sequence_index >= self.sequence_length {
                continue;
            }

            unsafe {
                // Same cache row placement as MatMulProj's qkv output.
                // 与 MatMulProj 的 qkv 输出采用相同的缓存行落位。
                let cache_row = next_sequence_index * self.batch_size + slice.batch_index;
                let qkv_row_ptr = self.qkv_ptr.ptr.add(cache_row * self.conv_dim);
                let state_ptr = self
                    .state_ptr
                    .ptr
                    .add(slice.batch_index * self.conv_dim * (self.kernel_size - 1));
                self.compute(
                    qkv_row_ptr as *const T,
                    self.weight_ptr.ptr,
                    state_ptr,
                    qkv_row_ptr,
                    channel_begin,
                    channel_end,
                );
                // Epilogue right after conv + SiLU, matching the reference
                // order: conv -> split -> l2norm(q, k) -> query scale.
                // 紧跟卷积 + SiLU 的 epilogue，与参考实现顺序一致：
                // conv -> split -> l2norm(q, k) -> query 缩放。
                self.norm_scale_qk(qkv_row_ptr, channel_begin, channel_end);
            }
        }
    }
}

impl<T> CausalConvTrait<T> for CausalConv1dSilu<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Sqrt,
{
    default fn compute(
        &self,
        _input_ptr: *const T,
        _weight_ptr: *const T,
        _state_ptr: *mut T,
        _output_ptr: *mut T,
        _channel_begin: usize,
        _channel_end: usize,
    ) {
        // TODO: compute logic, filled in later
    }

    // Per-head l2-norm (+ query scale) epilogue applied in place right
    // after conv + SiLU, mirroring the reference order
    // conv -> split -> l2norm(q, k) -> query scale:
    //   q_h *= (1 / sqrt(head_k_dim)) * rsqrt(sum(q_h^2) + eps)
    //   k_h *= rsqrt(sum(k_h^2) + eps)
    // Only heads fully inside [channel_begin, channel_end) are touched;
    // scheduling keeps block boundaries on head edges. The f16
    // specialization should accumulate the squared sum in fp32, like the
    // reference reduction.
    // 紧跟卷积 + SiLU 之后原地执行的逐头 l2 归一化（+ query 缩放）epilogue，
    // 与参考实现顺序一致：conv -> split -> l2norm(q, k) -> query 缩放：
    //   q_h *= (1 / sqrt(head_k_dim)) * rsqrt(sum(q_h^2) + eps)
    //   k_h *= rsqrt(sum(k_h^2) + eps)
    // 只处理完整落在 [channel_begin, channel_end) 内的头；
    // 调度保证块边界落在头边界上。f16 特化时应像参考实现的
    // reduction 一样用 fp32 累加平方和。
    default fn norm_scale_qk(&self, qkv_row_ptr: *mut T, channel_begin: usize, channel_end: usize) {
        // 与参考实现 l2norm 一致的容差。
        // Epsilon matching the reference l2norm.
        const EPS: f32 = 1e-6;

        let head_cols = self.head_k_dim;
        let q_scale = T::from_f32(1.0 / (head_cols as f32).sqrt());
        // (segment begin, segment end, extra factor): query folds the
        // attention scale into the normalization factor, key does not.
        // （段起点, 段终点, 额外系数）：query 把注意力缩放折进归一化系数，key 不折。
        let segments = [
            (0, self.key_dim, q_scale),
            (self.key_dim, 2 * self.key_dim, T::from_f32(1.0)),
        ];
        for (seg_begin, seg_end, scale) in segments {
            let begin = channel_begin.max(seg_begin);
            let end = channel_end.min(seg_end);
            if begin >= end {
                continue;
            }
            let first_head = (begin - seg_begin) / head_cols;
            let last_head = (end - seg_begin) / head_cols;
            for head in first_head..last_head {
                unsafe {
                    let head_ptr = qkv_row_ptr.add(seg_begin + head * head_cols);
                    let mut sum_sq = T::default();
                    for i in 0..head_cols {
                        let x = *head_ptr.add(i);
                        sum_sq = sum_sq + x * x;
                    }
                    let factor = scale / (sum_sq + T::from_f32(EPS)).sqrt();
                    for i in 0..head_cols {
                        let slot = head_ptr.add(i);
                        *slot = *slot * factor;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_conv1d_silu_construct_and_partition() {
        const M: usize = 4;
        const HEAD_K_DIM: usize = 2;
        const KEY_DIM: usize = 4; // two q heads + two k heads
        const CONV_DIM: usize = 2 * KEY_DIM + 4; // q, k and a value segment
        const KERNEL_SIZE: usize = 4;
        // qkv 缓存按 MatMulProj 的方式分配：sequence_length * batch_size 行。
        // qkv cache is allocated like MatMulProj: sequence_length * batch_size rows.
        const SEQUENCE_LENGTH: usize = M;
        const BATCH_SIZE: usize = 1;

        let weight_data: Vec<f32> = (0..CONV_DIM * KERNEL_SIZE)
            .map(|x| (x % 5) as f32 * 0.01)
            .collect();
        let mut state_data = vec![0.0f32; BATCH_SIZE * CONV_DIM * (KERNEL_SIZE - 1)];
        let mut qkv_data = vec![0.0f32; SEQUENCE_LENGTH * BATCH_SIZE * CONV_DIM];
        // Row 0 carries one non-zero q head and one non-zero k head so the
        // norm epilogue has something to normalize.
        // 第 0 行放一个非零 q 头和一个非零 k 头，供归一化 epilogue 检验。
        qkv_data[0] = 1.0;
        qkv_data[1] = 1.0;
        qkv_data[KEY_DIM] = 2.0;

        let operator = unsafe {
            CausalConv1dSilu::<f32>::new(
                qkv_data.as_mut_ptr(),
                weight_data.as_ptr(),
                state_data.as_mut_ptr(),
                KERNEL_SIZE,
                CONV_DIM,
                KEY_DIM,
                HEAD_K_DIM,
                SEQUENCE_LENGTH,
                BATCH_SIZE,
            )
        };
        assert_eq!(operator.kernel_size, KERNEL_SIZE);
        // 调度块必须按头对齐且不碎于 MIN_CHANNEL_BLOCK。
        // Scheduling blocks must stay head-aligned and SIMD friendly.
        assert_eq!(operator.head_block_size() % HEAD_K_DIM, 0);
        assert!(operator.head_block_size() >= 64);

        // compute is still empty, but the norm epilogue already runs: row
        // 0's q head takes l2norm + the 1 / sqrt(head_k_dim) scale, its k
        // head plain l2norm, everything else stays zero.
        // compute 仍为空，但归一化 epilogue 已生效：第 0 行的 q 头做
        // l2norm + 1 / sqrt(head_k_dim) 缩放，k 头只做 l2norm，其余保持 0。
        let thread_num = 4;
        let attention_list = [SequenceSlice {
            token_start_index: 0,
            batch_index: 0,
            next_sequence_index: 0,
            length: M,
            last_token_flag: true,
            lift_index: 0,
        }];
        for thread_id in 0..thread_num {
            operator.run(M, &attention_list, thread_num, thread_id);
        }
        let eps = 1e-6f32;
        let expected_q = (1.0 / (HEAD_K_DIM as f32).sqrt()) / (2.0 + eps).sqrt();
        let expected_k = 1.0 / (4.0 + eps).sqrt();
        assert!((qkv_data[0] - expected_q).abs() < 1e-6);
        assert!((qkv_data[1] - expected_q).abs() < 1e-6);
        assert!((qkv_data[KEY_DIM] - 2.0 * expected_k).abs() < 1e-6);
        assert!(qkv_data[KEY_DIM + 1] == 0.0);
        assert!(qkv_data[CONV_DIM..].iter().all(|&value| value == 0.0));
        assert!(state_data.iter().all(|&value| value == 0.0));
    }
}
