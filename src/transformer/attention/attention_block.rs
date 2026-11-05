use std::ops::{AddAssign, Neg, Sub};

use crate::config::TextConfig;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::FromNumber;
use crate::num_traits::NegInfinity;
use crate::num_traits::{Exp, Sigmoid, Sqrt};
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::attention_enum::AttentionEnum;

// Attention 运行时块骨架：持有 AttentionEnum（分支分派与核心注意力计算见
// attention_enum.rs），本层只保留全分支共享的 out_proj + residual 后处理。
pub(crate) struct AttentionBlock<T>
where
    T: Copy + PartialOrd,
{
    attention: AttentionEnum<T>,
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
    pub(crate) fn new(
        config: &TextConfig,
        attn_scope: &str,
        sequence_length: usize,
        batch_size: usize,
        layer_idx: usize,
    ) -> Self {
        Self {
            attention: AttentionEnum::new(
                config,
                attn_scope,
                sequence_length,
                batch_size,
                layer_idx,
            ),
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
        // enum 分支各自完成核心注意力计算，并带出后处理所需的差异参数
        // （输出权重、feature 维、MatMulParams、张量名）。
        let (attn_output, output_feature_dim, out_weight, params, output_tensor_name) = self
            .attention
            .forward(norm_hidden, position_embedding, decode_only_flag);

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
