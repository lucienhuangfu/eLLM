use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::init::matmul_params::MatMulParams;
use super::super::memory::cache::Cache;
use super::super::ptensor::tensor::Tensor;
use crate::compiler::operator::Operator;
// use super::mlp::MLP;
// use super::super::ptensor::linear::Linear;

#[derive(Clone)]
pub struct SparseMoeBlock<T> {
    hidden_size: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
    gate_weight: Tensor<T>,
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
        hidden_size: usize,
        intermediate_size: usize,
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
        residual: &Tensor<T>,
        tensor_name: String,
    ) -> Tensor<T> {
        let gate_output = hidden_states.matmul(
            &self.gate_weight,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            hidden_states.shape[0],
            self.scope_name.clone(),
        );

        let (topk_indices, topk_values) = gate_output
            .experts_softmax_norm(self.top_k, format!("{}.router_probs", self.scope_name));

        let nonlinear_product = hidden_states.experts_matmul_silu_mul_matmul(
            &self.experts_gate_weight,
            &self.experts_up_weight,
            &topk_indices,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            format!("{}.gate_up", self.scope_name),
        );

        let down_product = nonlinear_product.experts_matmul_merge_add(
            &self.experts_down_weight,
            &topk_indices,
            &topk_values,
            residual,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            format!("{}.down", self.scope_name),
        );
        down_product
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    use crate::memory::allocator::allocate_init;

    /*
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
    } */
}
