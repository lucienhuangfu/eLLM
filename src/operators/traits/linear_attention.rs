use crate::operators::attention::linear_attention::fused_gated_delta_rule::FusedScratchSlice;
use crate::runtime::SequenceSlice;

pub trait FusedGatedDeltaRuleTrait<T> {
    /// One (batch, head) task of the fused linear attention pipeline: the
    /// task's slices are processed strictly in order, their tokens chunked
    /// by chunk_size in a single interleaved loop. Each iteration =
    ///   (a) per token: four-way projection GEMV (qkv / z / b / a with the
    ///       beta / g epilogue) -> causal depthwise conv1d + SiLU on the
    ///       task's private conv_state window -> qk l2-norm (q scaled by
    ///       1/sqrt(head_k_dim)) into the scratch chunk buffers;
    ///   (b) chunk matricization: cum_decay = cumsum(g), pairwise_decay,
    ///       ut_system = (k_beta @ k^T) * pairwise_decay,
    ///       intra_chunk_attn = (q @ k^T) * pairwise_decay, forward
    ///       substitution new_values = ut^-1 v_beta, k_cumdecay = ut^-1
    ///       decayed_k_beta;
    ///   (c) one recurrence step: v_new = new_values - k_cumdecay @ S,
    ///       o = q @ S + intra_chunk_attn @ v_new,
    ///       S <- S * exp(cum_last) + k^T @ v_new -> gated RMSNorm
    ///       (w ⊙ o * rsqrt(mean(o^2) + eps) * silu(z)) -> write output.
    /// The updated S feeds the next iteration directly; no whole-sequence
    /// precomputation pass exists. head is the v head, its q/k head is
    /// head * num_k_heads / num_v_heads (repeat_interleave); a chunk_len
    /// == 1 fast path (decode) degenerates to the token-wise recurrence.
    /// 融合线性注意力流水线的单 (batch, head) 任务：严格按序处理该任务
    /// 的 slices，token 按 chunk_size 分块、单循环交错。每轮迭代 =
    ///   (a) 逐 token：四路投影 GEMV（qkv / z / b / a，含 beta / g
    ///       epilogue）-> 在任务私有 conv_state 窗口上因果 depthwise
    ///       卷积 + SiLU -> qk l2 归一化（q 乘 1/sqrt(head_k_dim)），
    ///       结果进 scratch chunk 缓冲；
    ///   (b) 本 chunk 矩阵化：cum_decay = cumsum(g)、pairwise_decay、
    ///       ut_system = (k_beta @ k^T) * pairwise_decay、
    ///       intra_chunk_attn = (q @ k^T) * pairwise_decay、前代求解
    ///       new_values = ut⁻¹ v_beta、k_cumdecay = ut⁻¹ decayed_k_beta；
    ///   (c) 本 chunk 一步递归：v_new = new_values - k_cumdecay @ S、
    ///       o = q @ S + intra_chunk_attn @ v_new、
    ///       S <- S * exp(cum_last) + k^T @ v_new -> gated RMSNorm
    ///       （w ⊙ o * rsqrt(mean(o²) + eps) * silu(z)）-> 写输出。
    /// 更新后的 S 直接供下一轮迭代，不存在全序列预计算 pass。
    /// head 为 v 头，其 q/k 头为 head * num_k_heads / num_v_heads
    /// （对应 repeat_interleave）；chunk_len == 1 快分支（decode）
    /// 退化为逐 token 递推公式。
    fn compute(
        &self,
        batch: usize,
        head: usize, // v head; q/k head = head * num_k_heads / num_v_heads
        slices: &[SequenceSlice],
        scratch: &mut FusedScratchSlice<'_, T>,
    );
}
