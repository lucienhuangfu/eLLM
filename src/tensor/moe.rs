use std::ops::{AddAssign, Neg, Sub};
use std::sync::atomic::AtomicUsize;

use crate::kernel::common::matmul_params::MatMulParams;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::NegInfinity;
use crate::num_traits::{Exp, Sigmoid, Sqrt};
use crate::operators::expert::expert_routing::{compact_moe_enabled, ExpertRouting};
use crate::operators::expert::{
    ExpertMatMulDown, ExpertMatMulSilu, ExpertMergeAdd, SharedExpertMLP, SharedExpertMergeAdd,
};
use crate::operators::operator::Operator;
use crate::operators::routing::{ExpertSoftmaxNorm, ExpertTopkNorm, MatMulSigmoid, TopKSoftmax};

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

        let operator = Operator::ExpertMergeAdd(ExpertMergeAdd::new(
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
        let token_count = routing.num_tokens;
        let output_shape = vec![token_count, num_experts_per_tok, down_weights.shape[1]];

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertMatMulDown(unsafe {
            ExpertMatMulDown::new(
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

        T::with_global(|pool| pool.remove(&down_weights.tensor_name));

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
    ) -> Self
    where
        T: Send + 'static,
    {
        // gate_weights [num_experts, intermediate_size, hidden_size]
        // Fixed layout: [num_experts, batch_size, intermediate_size].
        // Compact layout: [batch_size * top_k, intermediate_size].
        let token_count = self.row_count();
        let hidden_size = self.last_dim();
        let compact_moe = compact_moe_enabled();
        let output_shape = if compact_moe {
            vec![token_count * routing.num_topk, gate_weights.shape[1]]
        } else {
            vec![gate_weights.shape[0], token_count, gate_weights.shape[1]]
        };

        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let operator = Operator::ExpertMatMulSilu(unsafe {
            ExpertMatMulSilu::new(
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

        // Packed copies live in ExpertMatMulSilu; free originals.
        T::with_global(|pool| {
            pool.remove(&gate_weights.tensor_name);
            pool.remove(&up_weights.tensor_name);
        });

        Self::enqueue(operator);
        output_tensor
    }
}

// Unified shared+routed MoE-FFN path: a single fused MLP operator (gate/up + SiLU + down)
// followed by the shared merge-add. `shared_intermediate_size == 0` degenerates to routed-only.
// 统一 shared+routed MoE-FFN 路径：单个融合 MLP 算子（gate/up + SiLU + down）后接 shared
// merge-add。`shared_intermediate_size == 0` 退化为 routed-only。
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
    /// Fused gate/up + SiLU + down for shared + routed experts in one operator.
    /// Returns routed `[token, K, H]` and shared `[token, H]`; the intermediate nonlin
    /// never materializes (on-chip scratch only). When `shared_inter == 0` the shared
    /// weights are `None` (null pointers, never read) and Phase A is skipped → routed-only.
    #[allow(clippy::too_many_arguments)]
    pub fn experts_mlp_fused(
        &self,
        gate_weights: &Tensor<T>,                // [E, I, H]
        up_weights: &Tensor<T>,                  // [E, I, H]
        down_weights: &Tensor<T>,                // [E, H, I]
        shared_gate_weights: Option<&Tensor<T>>, // [Is, H]
        shared_up_weights: Option<&Tensor<T>>,   // [Is, H]
        shared_down_weights: Option<&Tensor<T>>, // [H, Is]
        routing: ExpertRouting<T>,
        num_experts_per_tok: usize,
        shared_inter: usize,
        decode_only_flag: bool,
        routed_scope_name: String,
        shared_scope_name: String,
    ) -> (Self, Self)
    where
        T: Send + 'static,
    {
        let token_count = self.row_count();
        let hidden_size = self.last_dim();
        let inter = gate_weights.shape[1];
        let num_experts = gate_weights.shape[0];

        let routed_shape = vec![token_count, num_experts_per_tok, hidden_size];
        let shared_shape = vec![token_count, hidden_size];
        let routed_tensor = Self::output_tensor(routed_shape, &routed_scope_name);
        let shared_tensor = Self::output_tensor(shared_shape, &shared_scope_name);

        let null_ptr: *const T = std::ptr::null();
        let operator = Operator::SharedExpertMLP(unsafe {
            SharedExpertMLP::new(
                self.data,
                gate_weights.data,
                up_weights.data,
                down_weights.data,
                shared_gate_weights
                    .map(|t| t.data as *const T)
                    .unwrap_or(null_ptr),
                shared_up_weights
                    .map(|t| t.data as *const T)
                    .unwrap_or(null_ptr),
                shared_down_weights
                    .map(|t| t.data as *const T)
                    .unwrap_or(null_ptr),
                routing,
                routed_tensor.data,
                shared_tensor.data,
                token_count,
                inter,
                shared_inter,
                hidden_size,
                num_experts,
                num_experts_per_tok,
                decode_only_flag,
            )
        });

        // Weights are read in place at run time (no packing), so keep them resident.
        // 权重在 run 时就地读取（不打包），因此保持常驻，不 pool.remove。
        Self::enqueue(operator);
        (routed_tensor, shared_tensor)
    }

    /// Merge residual + routed slots + (optionally) the gated shared-expert output.
    /// `shared_out`/`shared_gate_weight` are `None` for a routed-only model, in which case
    /// null pointers are passed and the operator skips the gated shared term.
    /// 合并 residual + routed slot +（可选）带门控的 shared expert 输出。routed-only 模型
    /// 传 `None`，此时用空指针，算子跳过带门控的 shared 项。
    #[allow(clippy::too_many_arguments)]
    pub fn experts_merge_add_fused(
        &self,
        residual: &Tensor<T>,                   // [token, H]
        shared_out: Option<&Tensor<T>>,         // [token, H]
        shared_gate_weight: Option<&Tensor<T>>, // [H]
        input_hidden: &Tensor<T>,               // [token, H]
        routing: ExpertRouting<T>,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        let output_shape = residual.shape.clone();
        let output_tensor = Self::output_tensor(output_shape, &scope_name);

        let null_ptr: *const T = std::ptr::null();
        let operator = Operator::SharedExpertMergeAdd(SharedExpertMergeAdd::new(
            self.data,
            residual.data,
            shared_out.map(|t| t.data as *const T).unwrap_or(null_ptr),
            shared_gate_weight
                .map(|t| t.data as *const T)
                .unwrap_or(null_ptr),
            input_hidden.data,
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
}

use super::storage::leaked_aligned_ptr;
use crate::mem_mgr::allocator::AlignedBox;

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

        let operator = Operator::ExpertSoftmaxNorm(ExpertSoftmaxNorm::new(
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
                gate_weight.tensor_name.clone(),
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

        let operator = Operator::ExpertTopkNorm(ExpertTopkNorm::new(
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

        // Keep B*K capacity because the current Top-K implementation may emit
        // repeated expert ids for a token. Reducing this to B corrupts the
        // adjacent routing buffers on real model output.
        let capacity_per_expert = num_tokens * num_topk;
        let mut index_tensor = AlignedBox::allocate_zero(num_experts * capacity_per_expert);
        let index_tensor = {
            let ptr = index_tensor.as_mut_ptr();
            std::mem::forget(index_tensor);
            ptr
        };
        let mut score_tensor = AlignedBox::allocate_zero(num_experts * capacity_per_expert);
        let score_tensor = {
            let ptr = score_tensor.as_mut_ptr();
            std::mem::forget(score_tensor);
            ptr
        };
        let mut topk_indices = AlignedBox::allocate_zero(num_tokens * num_topk);
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

        let operator = Operator::TopKSoftmax(
            TopKSoftmax::with_sampling(
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
            .with_input_thread_capacity(input_thread_capacity),
        );

        Self::enqueue(operator);
        (indice_ptr, value_tensor)
    }
}
