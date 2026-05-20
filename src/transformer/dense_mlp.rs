use std::ops::{AddAssign, Neg, Sub};

use crate::common::num_traits::NegInfinity;
use crate::common::num_traits::{Exp, Sigmoid, Sqrt};
use crate::mem_mgr::mem_pool::GlobalMemPool;

use super::super::common::matmul_params::MatMulParams;
use super::names::DenseMlpTensorNames;
use crate::tensor::{GlobalOperatorQueue, Tensor};

#[derive(Clone)]
pub struct DenseMlp<T>
where
    T: Copy + PartialOrd,
{
    gate_weight: Tensor<T>,
    up_weight: Tensor<T>,
    down_weight: Tensor<T>,
    scope_name: String,
}

impl<T> DenseMlp<T>
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
    pub fn new(hidden_size: usize, intermediate_size: usize, names: DenseMlpTensorNames) -> Self {
        Self {
            gate_weight: Tensor::zeros(vec![hidden_size, intermediate_size], names.gate_proj),
            up_weight: Tensor::zeros(vec![hidden_size, intermediate_size], names.up_proj),
            down_weight: Tensor::zeros(vec![intermediate_size, hidden_size], names.down_proj),
            scope_name: names.scope,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        _tensor_name: String,
    ) -> Tensor<T> {
        let gate_product = hidden_states.matmul(
            &self.gate_weight,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            hidden_states.shape[0],
            false,
            format!("{}.gate_proj.output", self.scope_name),
        );

        let up_product = hidden_states.matmul(
            &self.up_weight,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            hidden_states.shape[0],
            false,
            format!("{}.up_proj.output", self.scope_name),
        );

        let nonlinear_product =
            gate_product.add(&up_product, format!("{}.intermediate", self.scope_name));

        nonlinear_product.matmul_add(
            &self.down_weight,
            residual,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            format!("{}.output", self.scope_name),
        )
    }
}
