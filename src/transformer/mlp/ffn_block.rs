use std::ops::{AddAssign, Neg, Sub};

use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::NegInfinity;
use crate::num_traits::{Exp, Sigmoid, Sqrt};

use super::{DenseMlp, FfnKind, SparseMoe};
use crate::tensor::{GlobalOperatorQueue, Tensor};

// FFN 运行时块：配置蓝图 FfnKind 的已实例化对应物。仿照 SparseMoeRouter 范本——
// 私有 enum,由 new() 内一次 exhaustive match 从 config enum 构建,张量名在分支内派生。
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
    pub fn new(spec: &FfnKind, ffn_scope: &str, hidden_size: usize) -> Self {
        match spec {
            FfnKind::Dense { intermediate_size } => {
                FfnBlock::Dense(DenseMlp::new(hidden_size, *intermediate_size, ffn_scope))
            }
            FfnKind::SparseMoe {
                intermediate_size,
                num_experts,
                num_experts_per_tok,
                norm_topk_prob,
                router_scoring,
                use_routing_bias,
            } => FfnBlock::SparseMoe(SparseMoe::new(
                hidden_size,
                *intermediate_size,
                *num_experts,
                *num_experts_per_tok,
                *norm_topk_prob,
                router_scoring.clone(),
                *use_routing_bias,
                ffn_scope,
            )),
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
