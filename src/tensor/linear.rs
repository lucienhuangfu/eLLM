use std::ops::{AddAssign, Neg, Sub};

use crate::common::matmul_params::MatMulParams;
use crate::common::num_traits::{Exp, Sigmoid, Sqrt, NegInfinity};
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::linear::{Attention, MatMul, MatMul3, MatMulAdd};
use crate::operators::routing::MatMulTopK;
use crate::operators::operator::Operator;

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
    pub fn attention(
        &self,
        k_tensor: &Tensor<T>,
        v_tensor: &Tensor<T>,
        inverse_sqrt_head: T,
        decode_only_flag: bool,
        thread_num: usize,
        scope_name: String,
    ) -> Self {
        self.require_min_rank(4, "Tensor::attention query");
        k_tensor.require_min_rank(4, "Tensor::attention key");
        v_tensor.require_min_rank(4, "Tensor::attention value");

        let output_tensor = Self::output_tensor(self.shape.clone(), &scope_name);

        let operator = Operator::Attention(Attention::new(
            self.data,
            k_tensor.data,
            v_tensor.data,
            output_tensor.data,
            self.shape[0],
            self.shape[1],
            k_tensor.shape[1],
            k_tensor.shape[2],
            1,
            8,
            self.shape[2],
            inverse_sqrt_head,
            decode_only_flag,
            thread_num,
        ));

        Self::enqueue(operator);
        output_tensor
    }

    pub fn matmul(
        &self,
        tensor2: &Tensor<T>,
        params: MatMulParams,
        sequence_length: usize,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        self.require_min_rank(2, "Tensor::matmul input");
        tensor2.require_min_rank(2, "Tensor::matmul weight");

        let output_shape = vec![self.shape[0], tensor2.shape[0]];

        let output_to_kv = self.shape[0] > sequence_length;
        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = unsafe {
            Operator::MatMul(MatMul::new(
                self.data,
                tensor2.data,
                output_tensor.data,
                output_to_kv,
                params,
                self.shape[0],
                tensor2.shape[0],
                self.shape[1],
                decode_only_flag,
            ))
        };

        Self::enqueue(operator);
        output_tensor
    }

    pub fn matmul_add(
        &self,
        tensor2: &Tensor<T>,
        tensor3: &Tensor<T>,
        params: MatMulParams,
        tensor_name: String,
    ) -> Self {
        self.require_min_rank(2, "Tensor::matmul_add input");
        tensor2.require_min_rank(2, "Tensor::matmul_add weight");
        tensor3.require_min_rank(2, "Tensor::matmul_add residual");

        let output_shape = vec![self.shape[0], tensor2.shape[0]];

        let a_row = self.shape[0];
        let b_row = tensor2.shape[0];
        let column = self.shape[1];

        let output_tensor = Self::from_mem_pool(output_shape, tensor_name);

        let operator = Operator::MatMulAdd(unsafe {
            MatMulAdd::new(
                self.data,
                tensor2.data,
                tensor3.data,
                output_tensor.data,
                params,
                a_row,
                b_row,
                column,
            )
        });

        Self::enqueue(operator);
        output_tensor
    }

    pub fn matmul3(
        &self,
        q_weight: &Tensor<T>,
        k_weight: &Tensor<T>,
        v_weight: &Tensor<T>,
        position_embedding: &Tensor<T>,
        head_dim: usize,
        params: MatMulParams,
        scope_name: String,
    ) -> (Self, Self, Self) {
        self.require_min_rank(2, "Tensor::matmul3 input");
        q_weight.require_min_rank(2, "Tensor::matmul3 q_weight");
        k_weight.require_min_rank(2, "Tensor::matmul3 k_weight");
        v_weight.require_min_rank(2, "Tensor::matmul3 v_weight");
        position_embedding.require_min_rank(1, "Tensor::matmul3 position_embedding");

        let a_h_row = self.shape[0];
        let col = self.shape[1];

        let q_state = Self::from_mem_pool(
            vec![self.shape[0], q_weight.shape[0]],
            format!("{}.q_proj.output", scope_name),
        );

        let k_state = Self::from_mem_pool(
            vec![self.shape[0], k_weight.shape[0]],
            format!("{}.k_proj.output", scope_name),
        );

        let v_state = Self::from_mem_pool(
            vec![self.shape[0], v_weight.shape[0]],
            format!("{}.v_proj.output", scope_name),
        );

        let operator = Operator::MatMul3(MatMul3::new(
            self.data,
            q_weight.data,
            q_state.data,
            k_weight.data,
            k_state.data,
            v_weight.data,
            v_state.data,
            position_embedding.data,
            head_dim,
            a_h_row,
            col,
            q_weight.shape[0],
            k_weight.shape[0],
            params.a_row_step_macro,
            params.b_row_step_macro,
            params.column_step_macro,
            params.a_row_step_micro,
            params.b_row_step_micro,
        ));

        Self::enqueue(operator);
        (q_state, k_state, v_state)
    }

    pub fn matmul_local_topk(
        &self,
        tensor2: &Tensor<T>,
        params: MatMulParams,
        topk: usize,
        scope_name: String,
    ) -> (*const usize, Self) {
        self.require_min_rank(2, "Tensor::matmul_local_topk input");
        tensor2.require_min_rank(2, "Tensor::matmul_local_topk weight");

        let n = tensor2.shape[0];
        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let m = self.shape[0];
        let k = self.shape[1];
        let output_shape = vec![self.shape[0], thread_num * topk];

        let indice_ptr = leaked_aligned_ptr(output_shape.iter().product(), 0usize);

        let value_tensor =
            Self::from_mem_pool(output_shape, format!("{}.values.output", scope_name));

        let operator = unsafe {
            Operator::MatMulTopK(MatMulTopK::new(
                self.data,
                tensor2.data,
                indice_ptr,
                value_tensor.data,
                m,
                n,
                k,
                params.a_row_step_macro,
                params.b_row_step_macro,
                params.column_step_macro,
                params.a_row_step_micro,
                params.b_row_step_micro,
                m,
                topk,
            ))
        };

        Self::enqueue(operator);
        (indice_ptr, value_tensor)
    }
}
