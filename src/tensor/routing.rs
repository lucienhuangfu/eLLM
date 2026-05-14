use std::ops::{AddAssign, Neg, Sub};

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::operator::Operator;
use crate::operators::routing::{ExpertsSoftmaxNorm, ExpertsTopkNorm, MatMulSigmoid, TopKSoftmax};

use super::core::leaked_aligned_ptr;
use super::{GlobalOperatorQueue, Tensor};

impl<T> Tensor<T>
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
    pub fn softmax_norm(
        &self,
        num_experts: usize,
        num_experts_per_tok: usize,
        decode_only_flag: bool,
        _scope_name: String,
    ) -> (*mut bool, *mut bool, *mut T, *mut usize) {
        let experts_indicator = leaked_aligned_ptr(num_experts, false);
        let length = num_experts * self.shape[0];
        let indice_ptr = leaked_aligned_ptr(length, false);
        let weight_ptr = leaked_aligned_ptr(length, T::default());
        let topk_indices_ptr = leaked_aligned_ptr(num_experts_per_tok * self.shape[0], 0usize);

        let operator = Operator::ExpertsSoftmaxNorm(ExpertsSoftmaxNorm::new(
            self.data,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            topk_indices_ptr,
            self.shape[0],
            num_experts,
            num_experts_per_tok,
            decode_only_flag,
        ));
        Self::enqueue(operator);
        (experts_indicator, indice_ptr, weight_ptr, topk_indices_ptr)
    }

    pub fn sigmoid_gate(
        &self,
        gate_weight: &Tensor<T>,
        bias_tensor: Option<&Tensor<T>>,
        _decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        if let Some(bias_tensor) = bias_tensor {
            assert_eq!(
                bias_tensor.shape,
                vec![gate_weight.shape[0]],
                "sigmoid_gate bias shape mismatch"
            );
        }

        let output_shape = vec![self.shape[0], gate_weight.shape[0]];
        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 128,
            column_step_macro: 16,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };
        let operator = Operator::MatMulSigmoid(unsafe {
            MatMulSigmoid::new(
                self.data,
                gate_weight.data,
                bias_tensor.map(|tensor| tensor.data as *const T),
                output_tensor.data,
                params,
                self.shape[0],
                gate_weight.shape[0],
                self.shape[1],
                bias_tensor.is_some(),
            )
        });
        Self::enqueue(operator);
        output_tensor
    }

    pub fn topk_norm(
        &self,
        num_experts: usize,
        num_experts_per_tok: usize,
        decode_only_flag: bool,
        _scope_name: String,
    ) -> (*mut bool, *mut bool, *mut T, *mut usize) {
        let experts_indicator = leaked_aligned_ptr(num_experts, false);
        let length = num_experts * self.shape[0];
        let indice_ptr = leaked_aligned_ptr(length, false);
        let weight_ptr = leaked_aligned_ptr(length, T::default());
        let topk_indices_ptr = leaked_aligned_ptr(num_experts_per_tok * self.shape[0], 0usize);

        let operator = Operator::ExpertsTopkNorm(ExpertsTopkNorm::new(
            self.data,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            topk_indices_ptr,
            self.shape[0],
            num_experts,
            num_experts_per_tok,
            decode_only_flag,
        ));
        Self::enqueue(operator);
        (experts_indicator, indice_ptr, weight_ptr, topk_indices_ptr)
    }

    pub fn topk_softmax(
        &self,
        indices_ptr: *const usize,
        // sums_tensor: &Tensor<T>,
        output_sequences: *mut usize,
        batch_temperature: *mut T,
        num_topk: usize,
        eos_id: usize,
        scope_name: String,
    ) -> (*const usize, Self) {
        let output_shape = vec![self.shape[0], num_topk];
        let indice_ptr = leaked_aligned_ptr(output_shape.iter().product(), 0usize);

        let value_tensor =
            Self::from_mem_pool(output_shape, format!("{}.output_value.output", scope_name));

        let operator = Operator::TopKSoftmax(TopKSoftmax::new(
            indices_ptr,
            self.data,
            indice_ptr,
            value_tensor.data,
            output_sequences,
            batch_temperature,
            self.shape[0],
            num_topk,
            eos_id,
        ));

        Self::enqueue(operator);
        (indice_ptr, value_tensor)
    }
}
