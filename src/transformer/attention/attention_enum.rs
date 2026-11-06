use std::ops::{AddAssign, Neg, Sub};

use crate::config::TextConfig;
use crate::kernel::common::matmul_params::MatMulParams;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::FromNumber;
use crate::num_traits::NegInfinity;
use crate::num_traits::{Exp, Sigmoid, Sqrt};
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::full_attention::FullAttention;
use super::linear_attention::GatedDeltaAttention;
use crate::transformer::tensor_name::gated_delta_attention_tensor_names;

// Attention 运行时枚举：按 layer_types 原始字符串与 config 词表精确匹配分派——
// `linear_attention` → GatedDeltaNet，`full_attention` → 全因果注意力
// （滑窗已移除），其余 → panic。对应 Qwen3_5DecoderLayer 的 full_attention(self_attn) /
// linear_attention(linear_attn) 双分支。
// out_proj + residual 后处理骨架见 attention_block.rs 的 AttentionBlock。
pub(crate) enum AttentionEnum<T>
where
    T: Copy + PartialOrd,
{
    Full(FullAttention<T>),
    GatedDelta(GatedDeltaAttention<T>),
}

impl<T> AttentionEnum<T>
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
    pub(crate) fn new(
        config: &TextConfig,
        attn_scope: &str,
        sequence_length: usize,
        batch_size: usize,
        layer_idx: usize,
    ) -> Self {
        // 与 config 词表精确匹配的分派：layer_types[layer_idx] == "linear_attention" →
        // GatedDeltaNet；== "full_attention" → 全因果注意力；其余 → panic。
        // `validate()` 已保证清单长度等于层数，故 .get() 越界实际不可达，兜底仅为安全。
        match config.layer_type(layer_idx) {
            Some("linear_attention") => AttentionEnum::GatedDelta(GatedDeltaAttention::<T>::new(
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
            Some("full_attention") | None => {
                // 现有 family 全为因果注意力，is_causal 传 true。
                AttentionEnum::Full(FullAttention::<T>::new(
                    config,
                    sequence_length,
                    batch_size,
                    true,
                    attn_scope,
                ))
            }
            Some(other) => panic!("未知 layer_type: {other}"),
        }
    }

    // 执行核心注意力计算，并带出后处理（out_proj + residual）所需的差异参数
    // （输出权重、feature 维、MatMulParams、张量名）。
    pub(crate) fn forward(
        &self,
        norm_hidden: &Tensor<T>,
        position_embedding: &Tensor<T>,
        decode_only_flag: bool,
    ) -> (Tensor<T>, usize, &Tensor<T>, MatMulParams, String)
    where
        T: 'static,
    {
        // RoPE 全/滑窗注意力消费位置编码；GatedDeltaNet 线性注意力无位置编码，
        // 仅为统一签名而透传。
        match self {
            AttentionEnum::Full(attention) => {
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
            AttentionEnum::GatedDelta(gated_delta) => {
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
        }
    }
}
