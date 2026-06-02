use std::ops::{AddAssign, Neg, Sub};

use crate::operators::expert::expert_routing::ExpertRouting;
use crate::kernel::common::matmul_params::MatMulParams;
use crate::num_traits::{Exp, NegInfinity, Sigmoid, Sqrt};
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::super::names::SparseMoeTensorNames;
use super::router_sigmoid::SparseMoeSigmoidRouter;
use super::router_softmax::SparseMoeSoftmaxRouter;
use crate::transformer::config::RouterScoringKind;

#[derive(Clone)]
enum SparseMoeRouter<T>
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
    fn new(
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

    fn forward(&self, hidden_states: &Tensor<T>, decode_only_flag: bool) -> ExpertRouting<T> {
        match self {
            Self::Softmax(router) => router.forward(hidden_states, decode_only_flag),
            Self::Sigmoid(router) => router.forward(hidden_states, decode_only_flag),
        }
    }
}

#[derive(Clone)]
pub struct SparseMoe<T>
where
    T: Copy + PartialOrd,
{
    num_experts: usize,
    num_topk: usize,
    router: SparseMoeRouter<T>,
    experts_gate_weight: Tensor<T>,
    experts_up_weight: Tensor<T>,
    experts_down_weight: Tensor<T>,
    scope_name: String,
}

impl<T> SparseMoe<T>
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
    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        num_topk: usize,
        _norm_topk_prob: bool,
        router_scoring: RouterScoringKind,
        use_routing_bias: bool,
        names: SparseMoeTensorNames,
    ) -> Self {
        let scope_name = names.scope.clone();
        let gate_weight = Tensor::zeros(vec![num_experts, hidden_size], names.router_gate);
        let router_bias = if use_routing_bias {
            let bias_name = names
                .router_bias
                .expect("use_routing_bias is true but SparseMoeTensorNames.router_bias is None");
            Some(Tensor::zeros(vec![num_experts], bias_name))
        } else {
            None
        };

        Self {
            num_experts,
            num_topk,
            router: SparseMoeRouter::new(
                hidden_size,
                num_experts,
                num_topk,
                gate_weight,
                router_bias.clone(),
                router_scoring,
                scope_name.clone(),
            ),
            experts_gate_weight: Tensor::zeros(
                vec![num_experts, moe_intermediate_size, hidden_size],
                names.experts_gate_proj,
            ),
            experts_up_weight: Tensor::zeros(
                vec![num_experts, moe_intermediate_size, hidden_size],
                names.experts_up_proj,
            ),
            experts_down_weight: Tensor::zeros(
                vec![num_experts, hidden_size, moe_intermediate_size],
                names.experts_down_proj,
            ),
            scope_name,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        decode_only_flag: bool,
        tensor_name: String,
    ) -> Tensor<T> {
        let _ = tensor_name;
        let routing = self.router.forward(hidden_states, decode_only_flag);

        let nonlinear_product = hidden_states.experts_matmul_silu_mul_matmul(
            &self.experts_gate_weight,
            &self.experts_up_weight,
            routing,
            MatMulParams {
                a_row_step_macro: 3,
                b_row_step_macro: 128,
                column_step_macro: 128,
                a_row_step_micro: 3,
                b_row_step_micro: 32,
            },
            decode_only_flag,
            format!("{}.gate_up_proj.output", self.scope_name),
        );

        let down_product = nonlinear_product.experts_matmul_mul(
            &self.experts_down_weight,
            routing,
            self.num_topk,
            MatMulParams {
                a_row_step_macro: 3,
                b_row_step_macro: 128,
                column_step_macro: 128,
                a_row_step_micro: 3,
                b_row_step_micro: 32,
            },
            decode_only_flag,
            format!("{}.down_proj.output", self.scope_name),
        );

        down_product.experts_merge_add(
            residual,
            routing,
            decode_only_flag,
            format!("{}.output", self.scope_name),
        )
    }
}
