use std::ops::{AddAssign, Neg, Sub};

use crate::kernel::common::matmul_params::MatMulParams;
use crate::operators::expert::expert_routing::ExpertRouting;
// removed custom Sigmoid/Sqrt traits; use standard numeric ops instead
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, NegInfinity, Sigmoid, Sqrt};
use crate::tensor::{GlobalOperatorQueue, Tensor};

#[derive(Clone)]
pub(super) struct SparseMoeSoftmaxRouter<T>
where
    T: Copy + PartialOrd,
{
    num_experts: usize,
    num_topk: usize,
    gate_weight: Tensor<T>,
    scope_name: String,
}

impl<T> SparseMoeSoftmaxRouter<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + NegInfinity
        + Exp
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
        scope_name: String,
    ) -> Self {
        let expected_shape = vec![num_experts, hidden_size];
        assert_eq!(
            gate_weight.shape, expected_shape,
            "SparseMoeSoftmaxRouter gate weight shape mismatch"
        );

        Self {
            num_experts,
            num_topk,
            gate_weight,
            scope_name,
        }
    }

    pub(super) fn forward(
        &self,
        hidden_states: &Tensor<T>,
        decode_only_flag: bool,
    ) -> ExpertRouting<T> {
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

        gate_output.softmax_norm(
            self.num_experts,
            self.num_topk,
            decode_only_flag,
            format!("{}.router_probs", self.scope_name),
        )
    }
}
