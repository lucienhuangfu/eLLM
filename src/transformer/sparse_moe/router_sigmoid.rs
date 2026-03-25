use std::ops::{AddAssign, Neg, Sub};

use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::runtime::tensor::Tensor;

#[derive(Clone)]
pub(super) struct SparseMoeSigmoidRouter<T>
where
    T: Copy + PartialOrd,
{
    num_experts: usize,
    num_topk: usize,
    gate_weight: Tensor<T>,
    gate_bias: Option<Tensor<T>>,
    scope_name: String,
}

impl<T> SparseMoeSigmoidRouter<T>
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
        gate_bias: Option<Tensor<T>>,
        scope_name: String,
    ) -> Self {
        let expected_shape = vec![num_experts, hidden_size];
        assert_eq!(
            gate_weight.shape, expected_shape,
            "SparseMoeSigmoidRouter gate weight shape mismatch"
        );

        Self {
            num_experts,
            num_topk,
            gate_weight,
            gate_bias,
            scope_name,
        }
    }

    pub(super) fn forward(
        &self,
        hidden_states: &Tensor<T>,
        decode_only_flag: bool,
    ) -> (*mut bool, *mut bool, *mut T, *mut usize) {
        let gate_output = hidden_states.sigmoid_gate(
            &self.gate_weight,
            self.gate_bias.as_ref(),
            decode_only_flag,
            format!("{}.gate", self.scope_name),
        );

        gate_output.topk_norm(
            self.num_experts,
            self.num_topk,
            decode_only_flag,
            format!("{}.router_probs", self.scope_name),
        )
    }
}
