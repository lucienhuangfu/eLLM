use std::ops::{AddAssign, Neg, Sub};

use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, NegInfinity, Sigmoid, Sqrt};
use crate::operators::attention::linear_attention::FusedGatedDeltaRule;
use crate::operators::linear::MatMulProj;
use crate::operators::operator::Operator;

use super::{GlobalOperatorQueue, Tensor};

// Tensor builders for the GatedDeltaNet-style linear attention pipeline.
// The fused path is the one GatedDeltaAttention calls: fused_gated_delta_rule
// takes hidden straight to the attention output in a single operator.
// `matmul_proj` remains as a standalone builder for the input projection
// stage that other flows may reuse.
// The fused path keeps everything in per-thread scratch and uses the
// per-head conv_state layout [batch_size, num_v_heads, 2 * head_k_dim +
// head_v_dim, kernel_size - 1].
// GatedDeltaNet 线性注意力流水线的 Tensor 构建方法。
// 融合路径是 GatedDeltaAttention 实际调用的：fused_gated_delta_rule
// 单算子从 hidden 直出注意力输出。
// `matmul_proj` 作为单独的投影阶段 builder 保留，供其他流程复用。
// 融合路径的中间量全部住在线程私有 scratch，conv_state 采用逐头新布局
// [batch_size, num_v_heads, 2 * head_k_dim + head_v_dim,
// kernel_size - 1]。
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
