use std::ops::{AddAssign, Neg, Sub};

use crate::kernel::common::matmul_params::MatMulParams;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::model_spec::AttentionKind;
use crate::model_spec::TextConfig;
use crate::num_traits::FromNumber;
use crate::num_traits::NegInfinity;
use crate::num_traits::{Exp, Sigmoid, Sqrt};
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::full_attention::FullAttention;
use super::linear_attention::GatedDeltaAttention;
use super::tensor_name::gated_delta_attention_tensor_names;

// Attention 运行时块：配置蓝图 AttentionKind 的已实例化对应物。仿照 FfnBlock /
// SparseMoeRouter 范本——私有 enum,由 build() 内一次 exhaustive match 从 config enum
// 构建。对应 Qwen3_5DecoderLayer 的 full_attention(self_attn) / linear_attention
// (linear_attn) 双分支。
pub(crate) enum AttentionBlock<T>
where
    T: Copy + PartialOrd,
{
    Full(FullAttention<T>),
    GatedDelta(GatedDeltaAttention<T>),
}

impl<T> AttentionBlock<T>
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
        + FromNumber
        + AddAssign
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    pub(crate) fn build(
        config: &TextConfig,
        kind: AttentionKind,
        attn_scope: &str,
        sequence_length: usize,
        batch_size: usize,
        layer_idx: usize,
    ) -> Self {
        match kind {
            AttentionKind::Full | AttentionKind::SlidingWindow => AttentionBlock::Full(
                // 现有 family 全为因果注意力，is_causal 传 true。
                FullAttention::<T>::new(config, sequence_length, batch_size, true, attn_scope),
            ),
            AttentionKind::Linear => AttentionBlock::GatedDelta(GatedDeltaAttention::<T>::new(
                config.hidden_size,
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                config.linear_conv_kernel_dim,
                sequence_length,
                batch_size,
                gated_delta_attention_tensor_names(layer_idx),
            )),
        }
    }

    pub(crate) fn forward(
        &self,
        norm_hidden: &Tensor<T>,
        residual: &Tensor<T>,
        position_embedding: &Tensor<T>,
        decode_only_flag: bool,
    ) -> Tensor<T>
    where
        T: Send + 'static,
    {
        // RoPE 全/滑窗注意力消费位置编码；GatedDeltaNet 线性注意力无位置编码，
        // 仅为统一签名而透传。match 分支各自完成核心注意力计算，并带出
        // 后处理所需的差异参数（输出权重、feature 维、MatMulParams、张量名）。
        let (attn_output, output_feature_dim, out_weight, params, output_tensor_name) = match self {
            AttentionBlock::Full(attention) => {
                let attn_output =
                    attention.forward(norm_hidden, position_embedding, decode_only_flag);
                // q [chunk_size, head_num, group_num, head_dim] 的后两维即输出 feature 维
                let output_feature_dim = attn_output.shape[2] * attn_output.shape[3];
                (
                    attn_output,
                    output_feature_dim,
                    &attention.o_weight,
                    MatMulParams {
                        a_row_step_macro: 3,
                        b_row_step_macro: 256,
                        column_step_macro: 128,
                        a_row_step_micro: 3,
                        b_row_step_micro: 32,
                    },
                    format!("{}.output", attention.scope_name),
                )
            }
            AttentionBlock::GatedDelta(gated_delta) => {
                let attn_output =
                    gated_delta.forward(norm_hidden, position_embedding, decode_only_flag);
                (
                    attn_output,
                    gated_delta.value_dim,
                    &gated_delta.out_weight,
                    MatMulParams {
                        a_row_step_macro: 1,
                        b_row_step_macro: 64,
                        column_step_macro: 1,
                        a_row_step_micro: 1,
                        b_row_step_micro: 1,
                    },
                    format!("{}.output", gated_delta.scope_name),
                )
            }
        };

        // out_proj + residual.
        let output_sequence_length = attn_output.shape[0];
        let output_batch_size = attn_output.shape[1];
        let output_rows = output_sequence_length * output_batch_size;

        let view_attn_output = attn_output.view(vec![output_rows, output_feature_dim]);
        if decode_only_flag {
            view_attn_output.lift_vector();
        }
        let residual_hidden_size = *residual
            .shape
            .last()
            .expect("residual tensor must have at least one dimension");
        let view_residual = residual.view(vec![output_rows, residual_hidden_size]);

        // [sequence_length, batch_size, hidden_size]
        // matmul + add
        let output_2d = view_attn_output.matmul_add(
            out_weight,
            &view_residual,
            params,
            decode_only_flag,
            output_tensor_name,
        );
        output_2d.view(vec![
            output_sequence_length,
            output_batch_size,
            out_weight.shape[0],
        ])
    }
}
