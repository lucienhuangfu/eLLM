use std::ops::{AddAssign, Neg, Sub};
use std::sync::atomic::AtomicUsize;

use crate::kernel::common::matmul_params::MatMulParams;
use crate::num_traits::NegInfinity;
use crate::num_traits::{Exp, Sigmoid, Sqrt};
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::expert::expert_routing::ExpertRouting;
use crate::operators::expert::{ExpertsMatMulDown, ExpertsMatMulSilu, ExpertsMergeAdd};
use crate::operators::operator::Operator;
use crate::operators::routing::{ExpertsSoftmaxNorm, ExpertsTopkNorm, MatMulSigmoid, TopKSoftmax};

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

use crate::mem_mgr::allocator::AlignedBox;
use super::storage::leaked_aligned_ptr;

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
    ) -> ExpertRouting<T> {
        let token_count = self.row_count();
        let routing =
            unsafe { Self::allocate_expert_routing(num_experts, token_count, num_experts_per_tok) };

        let operator = Operator::ExpertsSoftmaxNorm(ExpertsSoftmaxNorm::new(
            self.data,
            routing,
            token_count,
            num_experts,
            num_experts_per_tok,
            decode_only_flag,
        ));
        Self::enqueue(operator);
        routing
    }

    pub fn sigmoid_gate(
        &self,
        gate_weight: &Tensor<T>,
        bias_tensor: Option<&Tensor<T>>,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        if let Some(bias_tensor) = bias_tensor {
            assert_eq!(
                bias_tensor.shape,
                vec![gate_weight.shape[0]],
                "sigmoid_gate bias shape mismatch"
            );
        }

        let token_count = self.row_count();
        let hidden_size = self.last_dim();
        let output_shape = vec![token_count, gate_weight.shape[0]];
        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 128,
            column_step_macro: 64,
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
                token_count,
                gate_weight.shape[0],
                hidden_size,
                bias_tensor.is_some(),
                decode_only_flag,
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
    ) -> ExpertRouting<T> {
        let token_count = self.row_count();
        let routing =
            unsafe { Self::allocate_expert_routing(num_experts, token_count, num_experts_per_tok) };

        let operator = Operator::ExpertsTopkNorm(ExpertsTopkNorm::new(
            self.data,
            routing,
            token_count,
            num_experts,
            num_experts_per_tok,
            decode_only_flag,
        ));
        Self::enqueue(operator);
        routing
    }

    unsafe fn allocate_expert_routing(
        num_experts: usize,
        num_tokens: usize,
        num_topk: usize,
    ) -> ExpertRouting<T> {
        let mut expert_counts_box = AlignedBox::<AtomicUsize>::allocate(num_experts);
        let expert_counts = expert_counts_box.as_mut_ptr();
        std::mem::forget(expert_counts_box);
        for e in 0..num_experts {
            std::ptr::write(expert_counts.add(e), AtomicUsize::new(0));
        }

        let capacity_per_expert = num_tokens * num_topk;
        let mut index_tensor = AlignedBox::allocate_init(num_experts * capacity_per_expert, 0usize);
        let index_tensor = {
            let ptr = index_tensor.as_mut_ptr();
            std::mem::forget(index_tensor);
            ptr
        };
        let mut score_tensor =
            AlignedBox::allocate_init(num_experts * capacity_per_expert, T::default());
        let score_tensor = {
            let ptr = score_tensor.as_mut_ptr();
            std::mem::forget(score_tensor);
            ptr
        };
        let mut topk_indices = AlignedBox::allocate_init(num_tokens * num_topk, 0usize);
        let topk_indices = {
            let ptr = topk_indices.as_mut_ptr();
            std::mem::forget(topk_indices);
            ptr
        };
        ExpertRouting {
            expert_counts: crate::operators::send_sync_ptr::MutPtr { ptr: expert_counts },
            index_tensor: crate::operators::send_sync_ptr::MutPtr { ptr: index_tensor },
            score_tensor: crate::operators::send_sync_ptr::MutPtr { ptr: score_tensor },
            topk_indices: crate::operators::send_sync_ptr::MutPtr { ptr: topk_indices },
            num_experts,
            num_tokens,
            num_topk,
            capacity_per_expert,
        }
    }

    pub fn topk_softmax(
        &self,
        indices_ptr: *const usize,
        output_sequences: *mut usize,
        batch_temperature: *mut T,
        sequence_stride: usize,
        input_top_k: usize,
        num_topk: usize,
        top_p: T,
        min_p: T,
        do_sample: bool,
        eos_ids: Vec<usize>,
        scope_name: String,
    ) -> (*const usize, Self) {
        let input_thread_capacity = self.shape[1].checked_div(input_top_k).unwrap_or(0).max(1);
        let output_shape = vec![self.shape[0], num_topk];
        let indice_ptr = leaked_aligned_ptr(output_shape.iter().product(), 0usize);

        let value_tensor =
            Self::from_mem_pool(output_shape, format!("{}.output_value.output", scope_name));

        let operator = Operator::TopKSoftmax(TopKSoftmax::with_sampling(
            indices_ptr,
            self.data,
            indice_ptr,
            value_tensor.data,
            output_sequences,
            batch_temperature,
            sequence_stride,
            input_top_k,
            num_topk,
            top_p,
            min_p,
            do_sample,
            eos_ids,
        )
        .with_input_thread_capacity(input_thread_capacity));

        Self::enqueue(operator);
        (indice_ptr, value_tensor)
    }
}
