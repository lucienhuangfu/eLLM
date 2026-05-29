use std::ops::{AddAssign, Neg, Sub};

use crate::operators::expert::expert_routing::ExpertRouting;
use crate::kernel::common::matmul_params::MatMulParams;
use crate::num_traits::NegInfinity;
use crate::num_traits::{Exp, Sigmoid, Sqrt};
use crate::mem_mgr::mem_pool::GlobalMemPool;
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
        routing: ExpertRouting<T>,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        let output_shape = residual.shape.clone();

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertsMergeAdd(ExpertsMergeAdd::new(
            self.data,
            residual.data,
            routing,
            output_tensor.data,
            1,
            self.row_count(),
            routing.num_experts,
            self.shape[1],
            self.last_dim(),
            false,
            decode_only_flag,
        ));

        Self::enqueue(operator);
        output_tensor
    }

    pub fn experts_matmul_mul(
        &self,
        down_weights: &Tensor<T>,
        routing: ExpertRouting<T>,
        num_experts_per_tok: usize,
        params: MatMulParams,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        // down_weights [num_experts, hidden_size, intermediate_size]
        // output [batch_size, num_experts_per_token, hidden_size]
        let token_count = self.shape[1];
        let output_shape = vec![token_count, num_experts_per_tok, down_weights.shape[1]];

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertsMatMulDown(unsafe {
            ExpertsMatMulDown::new(
                self.data,
                down_weights.data,
                routing,
                output_tensor.data,
                down_weights.shape[0],
                token_count,
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
        routing: ExpertRouting<T>,
        params: MatMulParams,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        // gate_weights [num_experts, intermediate_size, hidden_size]
        // output [num_experts, batch_size, intermediate_size]
        let token_count = self.row_count();
        let hidden_size = self.last_dim();
        let output_shape = vec![gate_weights.shape[0], token_count, gate_weights.shape[1]];

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertsMatMulSilu(unsafe {
            ExpertsMatMulSilu::new(
                self.data,
                gate_weights.data,
                up_weights.data,
                routing,
                output_tensor.data,
                token_count,
                gate_weights.shape[1],
                hidden_size,
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
