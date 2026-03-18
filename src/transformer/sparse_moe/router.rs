use std::ops::{AddAssign, Neg, Sub};

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::runtime::tensor::Tensor;

#[derive(Clone)]
pub(super) struct SparseMoeRouter<T>
where
    T: Copy + PartialOrd,
{
    num_experts: usize,
    num_topk: usize,
    gate_weight: Tensor<T>,
    scope_name: String,
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
        + AddAssign,
{
    pub(super) fn new(
        hidden_size: usize,
        num_experts: usize,
        num_topk: usize,
        gate_weight: Tensor<T>,
        scope_name: String,
    ) -> Self {
        let expected_shape = vec![num_experts, hidden_size];
        assert_eq!(
            gate_weight.shape, expected_shape,
            "SparseMoeRouter gate weight shape mismatch"
        );

        Self {
            num_experts,
            num_topk,
            gate_weight,
            scope_name,
        }
    }

    pub(super) fn route_tokens(
        &self,
        hidden_states: &Tensor<T>,
        decode_only_flag: bool,
    ) -> (*mut bool, *mut bool, *mut T, *mut usize) {
        let gate_output = hidden_states.matmul(
            &self.gate_weight,
            MatMulParams {
                a_row_step_macro: 3,
                b_row_step_macro: 128,
                column_step_macro: 16,
                a_row_step_micro: 3,
                b_row_step_micro: 32,
            },
            hidden_states.shape[0],
            decode_only_flag,
            format!("{}.gate", self.scope_name),
        );

        gate_output.experts_softmax_norm(
            self.num_experts,
            self.num_topk,
            decode_only_flag,
            format!("{}.router_probs", self.scope_name),
        )
    }
}