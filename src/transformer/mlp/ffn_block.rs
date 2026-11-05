use std::ops::{AddAssign, Neg, Sub};

use crate::config::TextConfig;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::NegInfinity;
use crate::num_traits::{Exp, Sigmoid, Sqrt};

use super::{DenseMlp, SparseMoe};
use crate::tensor::{GlobalOperatorQueue, Tensor};

// FFN 运行时块：逐层形态判断就在本构造点的行内 `if`，不经任何跨层蓝图中转，
// 逐字对应 `Qwen3MoeDecoderLayer.__init__` / `Qwen3NextDecoderLayer.__init__` 的
// `layer_idx not in mlp_only_layers and num_experts > 0 and (layer_idx + 1) %
// decoder_sparse_step == 0`。dense / MoE 的超参直接从 `TextConfig` 取，张量名在
// 分支内经 scope 派生。
pub enum FfnBlock<T>
where
    T: Copy + PartialOrd,
{
    Dense(DenseMlp<T>),
    SparseMoe(SparseMoe<T>),
}

impl<T> FfnBlock<T>
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
    pub fn new(config: &TextConfig, layer_idx: usize, ffn_scope: &str) -> Self {
        // A 类形态判断：三条件合取，行内展开。
        let sparse = !config.mlp_only_layers.contains(&layer_idx)
            && config.num_experts > 0
            && (layer_idx + 1) % config.decoder_sparse_step == 0;
        if sparse {
            FfnBlock::SparseMoe(SparseMoe::new(
                config.hidden_size,
                config.moe_intermediate_size,
                config.num_experts,
                config.num_experts_per_tok,
                &config.scoring_func,
                config.use_routing_bias,
                ffn_scope,
            ))
        } else {
            FfnBlock::Dense(DenseMlp::new(
                config.hidden_size,
                config.intermediate_size,
                ffn_scope,
            ))
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        decode_only_flag: bool,
        tensor_name: String,
    ) -> Tensor<T>
    where
        T: Send + 'static,
    {
        match self {
            FfnBlock::Dense(dense_mlp) => {
                dense_mlp.forward(hidden_states, residual, decode_only_flag, tensor_name)
            }
            FfnBlock::SparseMoe(sparse_moe) => {
                sparse_moe.forward(hidden_states, residual, decode_only_flag, tensor_name)
            }
        }
    }
}
