use std::ops::{AddAssign, Neg, Sub};

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::NegInfinity;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::common::num_traits::{Exp, Sigmoid, Sqrt};
use crate::operators::expert::{ExpertsMatMulDown, ExpertsMatMulSilu, ExpertsMergeAdd};
use crate::operators::operator::Operator;

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
    pub fn experts_merge_add(
        &self,
        residual: &Tensor<T>,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        num_experts: usize,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        self.require_min_rank(3, "Tensor::experts_merge_add input");
        residual.require_min_rank(2, "Tensor::experts_merge_add residual");

        // output [batch_size, hidden_size]
        let output_shape = vec![self.shape[0], self.shape[2]];

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertsMergeAdd(ExpertsMergeAdd::new(
            self.data,
            residual.data,
            experts_indicator,
            indice_ptr,
            output_tensor.data,
            1,
            num_experts,
            self.shape[0],
            self.shape[1],
            self.shape[2],
            false,
            decode_only_flag,
        ));

        Self::enqueue(operator);
        output_tensor
    }

    pub fn experts_matmul_mul(
        &self,
        down_weights: &Tensor<T>,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        weight_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        num_experts_per_tok: usize,
        params: MatMulParams,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        self.require_min_rank(3, "Tensor::experts_matmul_mul input");
        down_weights.require_min_rank(3, "Tensor::experts_matmul_mul weights");

        // down_weights [num_experts, hidden_size, intermediate_size]
        // output [batch_size, num_experts_per_token, hidden_size]
        let output_shape = vec![self.shape[1], num_experts_per_tok, down_weights.shape[1]];

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertsMatMulDown(unsafe {
            ExpertsMatMulDown::new(
                self.data,
                down_weights.data,
                experts_indicator,
                indice_ptr,
                weight_ptr,
                topk_indices_ptr,
                output_tensor.data,
                down_weights.shape[0],
                self.shape[1],
                down_weights.shape[2],
                down_weights.shape[1],
                num_experts_per_tok,
                params,
                decode_only_flag,
            )
        });

        Self::enqueue(operator);
        output_tensor
    }

    pub fn experts_matmul_silu_mul_matmul(
        &self,
        gate_weights: &Tensor<T>,
        up_weights: &Tensor<T>,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        // weight_ptr: *mut T,
        params: MatMulParams,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        self.require_min_rank(2, "Tensor::experts_matmul_silu_mul_matmul input");
        gate_weights.require_min_rank(3, "Tensor::experts_matmul_silu_mul_matmul gate");
        up_weights.require_min_rank(3, "Tensor::experts_matmul_silu_mul_matmul up");

        // gate_weights [num_experts, intermediate_size, hidden_size]
        // output [num_experts, batch_size, intermediate_size]
        let output_shape = vec![gate_weights.shape[0], self.shape[0], gate_weights.shape[1]];

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertsMatMulSilu(unsafe {
            ExpertsMatMulSilu::new(
                self.data,
                gate_weights.data,
                up_weights.data,
                experts_indicator,
                indice_ptr,
                output_tensor.data,
                self.shape[0],
                gate_weights.shape[1],
                self.shape[1],
                gate_weights.shape[0],
                params.a_row_step_macro,
                params.b_row_step_macro,
                params.column_step_macro,
                params.a_row_step_micro,
                params.b_row_step_micro,
                decode_only_flag,
            )
        });

        Self::enqueue(operator);
        output_tensor
    }
}
