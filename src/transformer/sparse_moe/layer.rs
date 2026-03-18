use std::ops::{AddAssign, Neg, Sub};
use std::rc::Rc;

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::runtime::tensor::{Tensor, TensorCtx};

use super::router::SparseMoeRouter;
use super::super::names::SparseMoeTensorNames;

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
        + AddAssign,
{
    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        num_topk: usize,
        _norm_topk_prob: bool,
        names: SparseMoeTensorNames,
        ctx: Rc<TensorCtx<T>>,
    ) -> Self {
        let scope_name = names.scope.clone();
        let gate_weight = ctx.zeros(vec![num_experts, hidden_size], names.router_gate);

        Self {
            num_experts,
            num_topk,
            router: SparseMoeRouter::new(
                hidden_size,
                num_experts,
                num_topk,
                gate_weight,
                scope_name.clone(),
            ),
            experts_gate_weight: ctx.zeros(
                vec![num_experts, moe_intermediate_size, hidden_size],
                names.experts_gate_proj,
            ),
            experts_up_weight: ctx.zeros(
                vec![num_experts, moe_intermediate_size, hidden_size],
                names.experts_up_proj,
            ),
            experts_down_weight: ctx.zeros(
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
        println!("Entering SparseMoe forward: {}", tensor_name);
        let (experts_indicator, indice_ptr, weight_ptr, topk_indices_ptr) =
            self.router.route_tokens(hidden_states, decode_only_flag);

        let nonlinear_product = hidden_states.experts_matmul_silu_mul_matmul(
            &self.experts_gate_weight,
            &self.experts_up_weight,
            experts_indicator,
            indice_ptr,
            MatMulParams {
                a_row_step_macro: 3,
                b_row_step_macro: 128,
                column_step_macro: 16,
                a_row_step_micro: 3,
                b_row_step_micro: 32,
            },
            decode_only_flag,
            format!("{}.gate_up", self.scope_name),
        );

        let down_product = nonlinear_product.experts_matmul_mul(
            &self.experts_down_weight,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            topk_indices_ptr,
            self.num_topk,
            MatMulParams {
                a_row_step_macro: 3,
                b_row_step_macro: 128,
                column_step_macro: 16,
                a_row_step_micro: 3,
                b_row_step_micro: 32,
            },
            decode_only_flag,
            format!("{}.down", self.scope_name),
        );

        down_product.experts_merge_add(
            residual,
            experts_indicator,
            indice_ptr,
            self.num_experts,
            decode_only_flag,
            format!("{}.merge", self.scope_name),
        )
    }
}