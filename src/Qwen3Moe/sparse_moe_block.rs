use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};
use crate::ptensor::matmul::MatMulParams


use super::super::memory::cache::Cache;
use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
// use super::mlp::MLP;
use crate::compiler::operator::Operator;

#[derive(Clone)]
pub struct SparseMoeBlock<T> {
    // sequence_chunk_size: usize,
    hidden_size: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
    gate: Linear<T>,
    experts_gate_weight: Tensor<T>,
    experts_up_weight: Tensor<T>,
    experts_down_weight: Tensor<T>,
    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> SparseMoeBlock<T>
where
    T: Copy + Default + Sub<Output = T> + Neg<Output = T> + Exp + NegInfinity + Sigmoid<T> + Sqrt,
{
    pub fn new(
        // sequence_chunk_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        // head_size: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = format!("{}.moe", parent_scope_name);
        Self {
            // sequence_chunk_size,
            hidden_size,
            num_experts,
            top_k,
            norm_topk_prob,
            gate_weight: Tensor::zeros(
                vec![hidden_size, num_experts],
                format!("{}.gate_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            experts_gate_weight: Tensor::zeros(
                vec![num_experts, hidden_size, intermediate_size],
                format!("{}.experts_gate_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            experts_up_weight: Tensor::zeros(
                vec![num_experts, hidden_size, intermediate_size],
                format!("{}.experts_up_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),

            experts_down_weight: Tensor::zeros(
                vec![num_experts, hidden_size, intermediate_size],
                format!("{}.experts_down_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            scope_name: scope_name,
            cache: cache,
            operator_queue: operator_queue,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        tensor_name: String,
    ) -> Tensor<T> {


        let (gate_output, sum) = self
            .gate
            .matmul_topk(hidden_states, format!("{}.gate_output", self.scope_name));

        let (topk_values, topk_indices) = gate_output.softmax_norm(format!("{}.router_probs", self.scope_name));
        // let (topk_values, topk_indices) = routing_weights.top_k(self.top_k, format!("{}.topk_indices", self.scope_name));
        // TODO: Implement the rest of the sparse MoE logic
        // For now, just return the first expert's output

        let nonlinear_product = hidden_states.experts_matmul_silu_mul_matmul(
            &self.experts_gate_weight,
            &self.experts_up_weight,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            &topk_indices,
            tensor_name,
        );

        let down_product = nonlinear_product.experts_matmul_merge_add(
            &self.down_weight,
            residual,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            &topk_indices,
            &topk_values,
            format!("{}.nonlinear_part1", self.scope_name),
        );
        down_product
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    use crate::memory::allocator::allocate_init;

    #[test]
    fn test_sparse_moe_block() {
        let position_window_size = 4;
        let batch_size = 32;
        let head_size = 128;

        let hidden_size = 8192;
        let intermediate_size = 4 * hidden_size;
        let num_experts = 8;
        let top_k = 2;
        let norm_topk_prob = 1;

        let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let sparse_moe = SparseMoeBlock::<f32>::new(
            position_window_size,
            hidden_size,
            num_experts,
            top_k,
            norm_topk_prob,
            intermediate_size,
            head_size,
            "model.layers.0",
            cache.clone(),
            operator_queue.clone(),
        );

        let shape = vec![position_window_size, batch_size, hidden_size];
        let input = Tensor::from_cache(
            shape.clone(),
            String::from("model.layers.0.input_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );
        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let output_tensor = sparse_moe.forward(
            &input,
            String::from("model.layers.0.output_tensor"),
            num_cpus::get(),
        );

        let thread_num: usize = num_cpus::get();
        for operator in output_tensor.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(1, 0, i);
            }
        }
    }

    /*
    #[test]
    fn test_feedforward_f16() {
        let batch_size = 32;
        let position_index = 1;
        let hidden_size = 8192;
        let head_size = 128;
        let hidden_dim = 4 * hidden_size;
        let multiple_of = 256;

        let cache: Rc<RefCell<Cache<f16>>> =
            Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let feedforward = FeedForward::<f16>::new(
            hidden_size,
            hidden_dim,
            head_size,
            multiple_of,
            "model.layers.0",
            cache.clone(),
            operator_queue.clone(),
        );

        let shape = vec![batch_size, hidden_size];
        let input = Tensor::from_cache(
            shape.clone(),
            String::from("model.layers.0.input_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );
        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let output_tensor = feedforward.forward(
            &input,
            String::from("model.layers.0.output_tensor"),
            num_cpus::get(),
        );

        let thread_num: usize = num_cpus::get();
        for operator in output_tensor.operator_queue.borrow().iter() {
            for i in 0..thread_num {
                operator.run(1, 0, i);
            }
        }

        /*
        let output_shape = vec![batch_size, hidden_size];
        let size = output_shape.iter().product();
        let mut result = vec![0.0; size];
        for i in 0..hidden_size {
            result[i] = hidden_dim as f32;
        }

        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, size) };
        assert_relative_eq!(output_slice, &result[..], max_relative = 1e-6);
         */
    }*/
}
