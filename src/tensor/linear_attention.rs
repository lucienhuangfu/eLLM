use std::ops::{AddAssign, Neg, Sub};

use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, NegInfinity, Sigmoid, Sqrt};
use crate::operators::attention::linear_attention::{FusedGatedDeltaRule, RecurrentGatedDeltaRule};
use crate::operators::conv::CausalConv1dSilu;
use crate::operators::linear::MatMulProj;
use crate::operators::operator::Operator;

use super::{GlobalOperatorQueue, Tensor};

// Tensor builders for the GatedDeltaNet-style linear attention pipelines.
// Two pipelines coexist in the library:
//   legacy: matmul_proj -> causal_conv_silu -> recurrent_gated_delta_rule
//   fused:  fused_gated_delta_rule (hidden -> attention output in one
//           operator; the legacy builders stay available but are no longer
//           called by GatedDeltaAttention)
// The qkv projection is a per-token cache laid out like the KV cache in
// matmul3 ([sequence_length, batch_size, cols]); the conv operator rewrites
// it in place and the recurrent operator consumes it read-only. The fused
// path keeps everything in per-thread scratch and uses the per-head
// conv_state layout [batch_size, num_v_heads, 2 * head_k_dim + head_v_dim,
// kernel_size - 1], which the legacy causal_conv_silu cannot consume.
// GatedDeltaNet 线性注意力流水线的 Tensor 构建方法，库内两条流水线并存：
//   旧路径：matmul_proj -> causal_conv_silu -> recurrent_gated_delta_rule
//   融合路径：fused_gated_delta_rule（hidden -> 注意力输出单算子完成；
//           旧 builder 保留可用，但 GatedDeltaAttention 不再调用）
// qkv 投影输出是与 matmul3 KV cache 同布局的逐 token 缓存
// （[sequence_length, batch_size, 列宽]），卷积算子原地改写该缓存，
// 递推算子只读消费。融合路径的中间量全部住在线程私有 scratch，
// conv_state 采用逐头新布局 [batch_size, num_v_heads,
// 2 * head_k_dim + head_v_dim, kernel_size - 1]，旧 causal_conv_silu
// 不能直接消费该布局。
impl<T> Tensor<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid
        + Sqrt
        + AddAssign
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    // Fused scheduling of the four input projections (qkv / z / b / a) over
    // the shared hidden_states input; each projection keeps its own weight
    // and output buffer. The qkv output is a per-token cache
    // [sequence_length, batch_size, qkv_cols]; z / beta / g are row-local
    // [input_rows, segment_cols].
    // 共享 hidden_states 输入的四路投影（qkv / z / b / a）融合调度；
    // 每路保留各自的权重与输出缓冲。qkv 输出为逐 token 缓存
    // [sequence_length, batch_size, qkv_cols]；z / beta / g 按行寻址
    // [input_rows, 段列宽]。
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_proj(
        &self,
        qkv_weight: &Tensor<T>,
        z_weight: &Tensor<T>,
        b_weight: &Tensor<T>,
        a_weight: &Tensor<T>,
        dt_bias: &Tensor<T>,
        a_log: &Tensor<T>,
        sequence_length: usize,
        batch_size: usize,
        scope_name: String,
    ) -> (Self, Self, Self, Self) {
        assert_eq!(
            a_weight.shape[0], b_weight.shape[0],
            "matmul_proj a/b segment head_cols mismatch"
        );

        let (active_sequence_length, active_batch_size) = if self.shape.len() >= 3 {
            (self.shape[0], self.shape[1])
        } else {
            (self.shape[0], 1)
        };
        let input_rows = active_sequence_length * active_batch_size;

        let qkv_cols = qkv_weight.shape[0];
        let value_dim = z_weight.shape[0];
        let head_cols = b_weight.shape[0];
        let reduction_cols = self.last_dim();

        let qkv_state = Self::from_mem_pool(
            vec![sequence_length, batch_size, qkv_cols],
            format!("{}.qkv_proj.output", scope_name),
        );
        let z_state = Self::from_mem_pool(
            vec![input_rows, value_dim],
            format!("{}.z_proj.output", scope_name),
        );
        let beta_state = Self::from_mem_pool(
            vec![input_rows, head_cols],
            format!("{}.beta_proj.output", scope_name),
        );
        let g_state = Self::from_mem_pool(
            vec![input_rows, head_cols],
            format!("{}.g_proj.output", scope_name),
        );

        let operator = unsafe {
            Operator::MatMulProj(MatMulProj::new(
                self.data,
                qkv_weight.data,
                z_weight.data,
                b_weight.data,
                a_weight.data,
                qkv_state.data,
                z_state.data,
                beta_state.data,
                g_state.data,
                dt_bias.data,
                a_log.data,
                qkv_cols,
                value_dim,
                head_cols,
                sequence_length,
                batch_size,
                input_rows,
                reduction_cols,
            ))
        };

        Self::enqueue(operator);
        (qkv_state, z_state, beta_state, g_state)
    }

    // Fused depthwise causal conv1d + SiLU + rolling state update with the
    // qk-norm epilogue, rewriting the qkv cache from matmul_proj in place.
    // state is the per-batch rolling window
    // [batch_size, conv_dim, kernel_size - 1] owned by the layer cache.
    // 融合 depthwise 因果卷积 + SiLU + 滚动状态更新（含 qk-norm epilogue），
    // 原地改写 matmul_proj 的 qkv 缓存。state 是层缓存持有的逐 batch
    // 滚动窗口 [batch_size, conv_dim, kernel_size - 1]。
    pub fn causal_conv_silu(
        &self,
        weight: &Tensor<T>,
        state: &Tensor<T>,
        key_dim: usize,
        head_k_dim: usize,
        sequence_length: usize,
        batch_size: usize,
    ) {
        let conv_dim = self.last_dim();
        let kernel_size = weight.last_dim();

        let operator = unsafe {
            Operator::CausalConv1dSilu(CausalConv1dSilu::new(
                self.data,
                weight.data,
                state.data,
                kernel_size,
                conv_dim,
                key_dim,
                head_k_dim,
                sequence_length,
                batch_size,
            ))
        };

        Self::enqueue(operator);
    }

    // Fused single-step gated delta rule recurrence over the qkv cache
    // (already convolved + normalized upstream); state is the per-layer
    // recurrent cache [batch_size, num_v_heads, head_k_dim, head_v_dim],
    // updated in place. Returns the attention output
    // [sequence_length, batch_size, value_dim].
    // 在已完成卷积与归一化的 qkv 缓存上执行单步 gated delta rule 递推。
    // state 是逐层递推缓存 [batch_size, num_v_heads, head_k_dim, head_v_dim]，
    // 原地更新。返回注意力输出 [sequence_length, batch_size, value_dim]。
    #[allow(clippy::too_many_arguments)]
    pub fn recurrent_gated_delta_rule(
        &self,
        g_tensor: &Tensor<T>,
        beta_tensor: &Tensor<T>,
        state: &Tensor<T>,
        key_dim: usize,
        sequence_length: usize,
        batch_size: usize,
        scope_name: String,
    ) -> Self {
        let num_v_heads = state.shape[1];
        let head_k_dim = state.shape[2];
        let head_v_dim = state.shape[3];
        let num_k_heads = key_dim / head_k_dim;
        // Width of the v segment inside the qkv row: num_v_heads * head_v_dim.
        // qkv 行内 v 段的宽度：num_v_heads * head_v_dim。
        let value_dim = num_v_heads * head_v_dim;

        let output_tensor =
            Self::output_tensor(vec![sequence_length, batch_size, value_dim], &scope_name);

        let operator = unsafe {
            Operator::RecurrentGatedDeltaRule(RecurrentGatedDeltaRule::new(
                self.data,
                g_tensor.data,
                beta_tensor.data,
                state.data,
                output_tensor.data,
                key_dim,
                value_dim,
                head_k_dim,
                head_v_dim,
                num_k_heads,
                num_v_heads,
                sequence_length,
                batch_size,
            ))
        };

        Self::enqueue(operator);
        output_tensor
    }

    // Fused linear attention: hidden -> four-way projection -> causal
    // conv1d + SiLU -> qk-norm -> chunked gated delta rule (single
    // interleaved loop) -> gated RMSNorm -> attention output
    // [sequence_length, batch_size, value_dim]. conv_state must use the
    // per-head layout [batch_size, num_v_heads, 2 * head_k_dim +
    // head_v_dim, kernel_size - 1]; recurrent_state
    // [batch_size, num_v_heads, head_k_dim, head_v_dim] is updated in
    // place. Tasks are (batch, v_head) pairs, batch_size * num_v_heads in
    // total; the scratch pool is sized for thread_num lanes.
    // 融合线性注意力：hidden -> 四路投影 -> 因果卷积 + SiLU ->
    // qk 归一化 -> chunk 化 gated delta rule（单循环交错）->
    // gated RMSNorm -> 注意力输出 [sequence_length, batch_size,
    // value_dim]。conv_state 必须采用逐头新布局 [batch_size,
    // num_v_heads, 2 * head_k_dim + head_v_dim, kernel_size - 1]；
    // recurrent_state [batch_size, num_v_heads, head_k_dim, head_v_dim]
    // 原地更新。任务为 (batch, v_head) 对，共 batch_size * num_v_heads
    // 个；scratch 池按 thread_num 条通道预分配。
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gated_delta_rule(
        &self,
        qkv_weight: &Tensor<T>,
        z_weight: &Tensor<T>,
        b_weight: &Tensor<T>,
        a_weight: &Tensor<T>,
        dt_bias: &Tensor<T>,
        a_log: &Tensor<T>,
        conv_weight: &Tensor<T>,
        norm_weight: &Tensor<T>,
        conv_state: &Tensor<T>,
        recurrent_state: &Tensor<T>,
        chunk_size: usize,
        thread_num: usize,
        sequence_length: usize,
        batch_size: usize,
        scope_name: String,
    ) -> Self {
        let hidden_size = self.last_dim();
        let conv_dim = qkv_weight.shape[0];
        let value_dim = z_weight.shape[0];
        let num_v_heads = recurrent_state.shape[1];
        let head_k_dim = recurrent_state.shape[2];
        let head_v_dim = recurrent_state.shape[3];
        let key_dim = (conv_dim - value_dim) / 2;
        let num_k_heads = key_dim / head_k_dim;
        debug_assert_eq!(key_dim % head_k_dim, 0);
        let kernel_size = conv_weight.last_dim();
        let input_rows = sequence_length * batch_size;

        let output_tensor =
            Self::output_tensor(vec![sequence_length, batch_size, value_dim], &scope_name);

        let operator = unsafe {
            Operator::FusedGatedDeltaRule(FusedGatedDeltaRule::new(
                self.data,
                qkv_weight.data,
                z_weight.data,
                b_weight.data,
                a_weight.data,
                conv_weight.data,
                norm_weight.data,
                dt_bias.data,
                a_log.data,
                conv_state.data,
                recurrent_state.data,
                output_tensor.data,
                hidden_size,
                key_dim,
                value_dim,
                head_k_dim,
                head_v_dim,
                num_k_heads,
                num_v_heads,
                kernel_size,
                chunk_size,
                sequence_length,
                batch_size,
                input_rows,
                hidden_size,
                thread_num,
            ))
        };

        Self::enqueue(operator);
        output_tensor
    }
}
