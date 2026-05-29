use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::sync::atomic::AtomicUsize;

use crate::kernel::common::matmul_params::MatMulParams;
use crate::mem_mgr::allocator::AlignedBox;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, NegInfinity, Sigmoid, Sqrt};
use crate::operators::expert::expert_routing::ExpertRouting;
use crate::operators::linear::{Attention, MatMul, MatMul3, MatMulAdd};
use crate::operators::moe::{ExpertMatMulDown, ExpertMatMulSilu, ExpertMergeAdd};
use crate::operators::movement::LiftVector;
use crate::operators::operator::Operator;
use crate::operators::routing::{
    ExpertSoftmaxNorm, ExpertTopkNorm, MatMulSigmoid, MatMulTopK, TopKSoftmax,
};
use crate::operators::transform::{AddRMSZipMap, AddZipMap, LookupRMSMap, RMSMap, SigmoidMap};

use super::core::leaked_aligned_ptr;
use super::{GlobalOperatorQueue, Tensor};

impl<T> Tensor<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Add<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
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
        sequence_length: usize,
        batch_size: usize,
        inverse_sqrt_head: T,
        decode_only_flag: bool,
        thread_num: usize,
        scope_name: String,
    ) -> Self {
        let output_tensor = Self::output_tensor(self.shape.clone(), &scope_name);

        let operator = Operator::Attention(Attention::new(
            self.data,
            k_tensor.data,
            v_tensor.data,
            output_tensor.data,
            sequence_length,
            batch_size,
            self.shape[2],
            k_tensor.shape[1],
            k_tensor.strides[0],
            k_tensor.strides[1],
            k_tensor.strides[2],
            v_tensor.strides[0],
            v_tensor.strides[1],
            v_tensor.strides[2],
            1,
            8,
            self.shape[3],
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
        decode_only_flag: bool,
        tensor_name: String,
    ) -> Self {
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
                decode_only_flag,
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
        sequence_length: usize,
        batch_size: usize,
        kv_head_num: usize,
        group_num: usize,
        head_dim: usize,
        params: MatMulParams,
        scope_name: String,
    ) -> (Self, Self, Self) {
        let (active_sequence_length, active_batch_size, hidden_size) = if self.shape.len() >= 3 {
            (self.shape[0], self.shape[1], self.shape[2])
        } else {
            (self.shape[0], 1, self.shape[1])
        };
        let output_rows = active_sequence_length * active_batch_size;

        let q_state = Self::from_mem_pool(
            vec![active_sequence_length, active_batch_size, q_weight.shape[0]],
            format!("{}.q_proj.output", scope_name),
        );

        let k_state = Self::from_mem_pool(
            vec![sequence_length, batch_size, k_weight.shape[0]],
            format!("{}.k_proj.output", scope_name),
        );

        let v_state = Self::from_mem_pool(
            vec![sequence_length, batch_size, v_weight.shape[0]],
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
            sequence_length,
            batch_size,
            kv_head_num,
            group_num,
            head_dim,
            output_rows,
            hidden_size,
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
        thread_num: usize,
        scope_name: String,
    ) -> (*const usize, Self) {
        let n = tensor2.shape[0];
        let m = self.shape[0];
        let k = self.shape[1];
        let thread_num = thread_num.max(1);
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
                topk,
                thread_num,
                m,
            ))
        };

        Self::enqueue(operator);
        (indice_ptr, value_tensor)
    }
}

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
    pub fn add(&self, b_tensor: &Tensor<T>, decode_only_flag: bool, tensor_name: String) -> Self {
        let output_tensor = Self::from_mem_pool(self.shape.clone(), tensor_name);
        let operator = Operator::AddZipMap(AddZipMap::new(
            self.data,
            b_tensor.data,
            output_tensor.data,
            self.shape[1],
            self.shape[2],
            decode_only_flag,
        ));
        Self::enqueue(operator);
        output_tensor
    }

    pub fn add_rms(
        &self,
        b_tensor: &Tensor<T>,
        _weight: *const T,
        eps: T,
        tensor_name: String,
    ) -> Self {
        let output_tensor = Self::from_mem_pool(self.shape.clone(), tensor_name);

        let operator = Operator::AddRMSZipMap(AddRMSZipMap::new(
            self.data,
            b_tensor.data,
            output_tensor.data,
            self.shape[1],
            eps,
        ));
        Self::enqueue(operator);
        output_tensor
    }

    pub fn sigmoid(&self, tensor_name: String) -> Self {
        let output_tensor = Self::from_mem_pool(self.shape.clone(), tensor_name);
        let operator = Operator::SigmoidMap(SigmoidMap::new(
            self.data,
            output_tensor.data,
            self.element_count(),
        ));
        Self::enqueue(operator);
        output_tensor
    }

    pub fn lookup_rms(
        sequences_ptr: *const usize,
        word_embedding: &Tensor<T>,
        token_capacity: usize,
        sequence_stride: usize,
        eps: T,
        scope_name: String,
    ) -> (Self, Self) {
        let output_hidden_tensor = Self::from_mem_pool(
            vec![token_capacity, word_embedding.shape[1]],
            format!("{}.output_hidden", scope_name),
        );

        let output_normal_tensor = Self::from_mem_pool(
            vec![token_capacity, word_embedding.shape[1]],
            format!("{}.output_normal", scope_name),
        );

        let operator = Operator::LookupRMSMap(LookupRMSMap::new(
            sequences_ptr,
            word_embedding.data,
            output_hidden_tensor.data,
            output_normal_tensor.data,
            sequence_stride,
            word_embedding.shape[1],
            eps,
        ));

        Self::enqueue(operator);
        (output_hidden_tensor, output_normal_tensor)
    }

    pub fn lift_vector(&self) {
        let row_len = self.shape.iter().skip(1).product();
        let operator = Operator::LiftVector(LiftVector::new(self.data, row_len));
        Self::enqueue(operator);
    }

    pub fn rms(&self, eps: T, decode_only_flag: bool, scope_name: String) -> Self {
        let output_tensor = Self::output_tensor(self.shape.clone(), &scope_name);

        let operator = Operator::RMSMap(RMSMap::new(
            self.data,
            output_tensor.data,
            self.shape[1],
            eps,
            decode_only_flag,
        ));
        Self::enqueue(operator);
        output_tensor
    }
}

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
        let routing = unsafe {
            Self::allocate_expert_routing(num_experts, self.shape[0], num_experts_per_tok)
        };

        let operator = Operator::ExpertSoftmaxNorm(ExpertSoftmaxNorm::new(
            self.data,
            routing,
            self.shape[0],
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
        let routing = unsafe {
            Self::allocate_expert_routing(num_experts, self.shape[0], num_experts_per_tok)
        };

        let operator = Operator::ExpertTopkNorm(ExpertTopkNorm::new(
            self.data,
            routing,
            self.shape[0],
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
        let expert_counts_box = AlignedBox::<AtomicUsize>::allocate(num_experts);
        let expert_counts = expert_counts_box.as_mut_ptr();
        std::mem::forget(expert_counts_box);
        for e in 0..num_experts {
            std::ptr::write(expert_counts.add(e), AtomicUsize::new(0));
        }

        let capacity_per_expert = num_tokens * num_topk;
        let index_tensor = AlignedBox::allocate_init(num_experts * capacity_per_expert, 0usize);
        let index_tensor = {
            let ptr = index_tensor.as_mut_ptr();
            std::mem::forget(index_tensor);
            ptr
        };
        let score_tensor =
            AlignedBox::allocate_init(num_experts * capacity_per_expert, T::default());
        let score_tensor = {
            let ptr = score_tensor.as_mut_ptr();
            std::mem::forget(score_tensor);
            ptr
        };
        let topk_indices = AlignedBox::allocate_init(num_tokens * num_topk, 0usize);
        let topk_indices = {
            let ptr = topk_indices.as_mut_ptr();
            std::mem::forget(topk_indices);
            ptr
        };
        ExpertRouting {
            expert_counts: crate::common::send_sync_ptr::MutPtr { ptr: expert_counts },
            index_tensor: crate::common::send_sync_ptr::MutPtr { ptr: index_tensor },
            score_tensor: crate::common::send_sync_ptr::MutPtr { ptr: score_tensor },
            topk_indices: crate::common::send_sync_ptr::MutPtr { ptr: topk_indices },
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
        num_topk: usize,
        top_k_simd: usize,
        thread_num: usize,
        top_p: T,
        min_p: T,
        do_sample: bool,
        eos_token_id_list: Vec<usize>,
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
            sequence_stride,
            num_topk,
            top_k_simd,
            thread_num,
            top_p,
            min_p,
            do_sample,
            eos_token_id_list,
        ));

        Self::enqueue(operator);
        (indice_ptr, value_tensor)
    }
}

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
        let output_shape = vec![self.shape[0], self.shape[2]];

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertMergeAdd(ExpertMergeAdd::new(
            self.data,
            residual.data,
            routing,
            output_tensor.data,
            1,
            self.shape[0],
            routing.num_experts,
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
        routing: ExpertRouting<T>,
        num_experts_per_tok: usize,
        params: MatMulParams,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        let output_shape = vec![self.shape[1], num_experts_per_tok, down_weights.shape[1]];

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertMatMulDown(unsafe {
            ExpertMatMulDown::new(
                self.data,
                down_weights.data,
                routing,
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
        routing: ExpertRouting<T>,
        params: MatMulParams,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        let output_shape = vec![gate_weights.shape[0], self.shape[0], gate_weights.shape[1]];

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertMatMulSilu(unsafe {
            ExpertMatMulSilu::new(
                self.data,
                gate_weights.data,
                up_weights.data,
                routing,
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
