use std::ops::{AddAssign, Neg, Sub};

use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, NegInfinity, Sigmoid, Sqrt};
use crate::operators::expert::expert_routing::ExpertRouting;
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::router_sigmoid::SparseMoeSigmoidRouter;
use super::router_softmax::SparseMoeSoftmaxRouter;

#[derive(Clone)]
pub(crate) enum SparseMoeRouter<T>
where
    T: Copy + PartialOrd,
{
    Softmax(SparseMoeSoftmaxRouter<T>),
    Sigmoid(SparseMoeSigmoidRouter<T>),
}

impl<T> SparseMoeRouter<T>
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
    pub(crate) fn new(
        hidden_size: usize,
        num_experts: usize,
        num_topk: usize,
        gate_weight: Tensor<T>,
        gate_bias: Option<Tensor<T>>,
        scoring_func: &str,
        scope_name: String,
    ) -> Self {
        // 单点字符串分派，与 AttentionBlock::new 的 layer_types 分派同形：
        // 含 "sigmoid" → Sigmoid router，其余（softmax / 未识别）兜底 Softmax。
        // `TextConfig::validate()` 已在加载期归一化并限定词表，此处不再 lowercase。
        if scoring_func.contains("sigmoid") {
            Self::Sigmoid(SparseMoeSigmoidRouter::new(
                hidden_size,
                num_experts,
                num_topk,
                gate_weight,
                gate_bias,
                scope_name,
            ))
        } else {
            Self::Softmax(SparseMoeSoftmaxRouter::new(
                hidden_size,
                num_experts,
                num_topk,
                gate_weight,
                scope_name,
            ))
        }
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<T>,
        decode_only_flag: bool,
    ) -> ExpertRouting<T> {
        match self {
            Self::Softmax(router) => router.forward(hidden_states, decode_only_flag),
            Self::Sigmoid(router) => router.forward(hidden_states, decode_only_flag),
        }
    }
}
