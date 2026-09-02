use std::ops::{AddAssign, Neg, Sub};

use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, NegInfinity, Sigmoid, Sqrt};
use crate::operators::expert::expert_routing::ExpertRouting;
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::router_sigmoid::SparseMoeSigmoidRouter;
use super::router_softmax::SparseMoeSoftmaxRouter;
use crate::model_family::config::RouterScoringKind;

#[derive(Clone)]
pub(super) enum SparseMoeRouter<T>
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
    pub(super) fn new(
        hidden_size: usize,
        num_experts: usize,
        num_topk: usize,
        gate_weight: Tensor<T>,
        gate_bias: Option<Tensor<T>>,
        router_scoring: RouterScoringKind,
        scope_name: String,
    ) -> Self {
        match router_scoring {
            RouterScoringKind::Softmax => Self::Softmax(SparseMoeSoftmaxRouter::new(
                hidden_size,
                num_experts,
                num_topk,
                gate_weight,
                scope_name,
            )),
            RouterScoringKind::Sigmoid => Self::Sigmoid(SparseMoeSigmoidRouter::new(
                hidden_size,
                num_experts,
                num_topk,
                gate_weight,
                gate_bias,
                scope_name,
            )),
        }
    }

    pub(super) fn forward(
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
