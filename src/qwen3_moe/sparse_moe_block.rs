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
    num_topk: usize,
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
    T: Copy
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid<T>
        + Sqrt
        + AddAssign,
{
    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        num_topk: usize,
        norm_topk_prob: bool,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = format!("{}.mlp", parent_scope_name);
        Self {
            // sequence_chunk_size,
            hidden_size,
            num_experts,
            num_topk,
            norm_topk_prob,
            gate_weight: Tensor::zeros(
                vec![num_experts, hidden_size],
                format!("{}.gate.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            experts_gate_weight: Tensor::zeros(
                vec![num_experts, moe_intermediate_size, hidden_size],
                format!("{}.experts.gate_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            experts_up_weight: Tensor::zeros(
                vec![num_experts, moe_intermediate_size, hidden_size],
                format!("{}.experts.up_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),

            experts_down_weight: Tensor::zeros(
                vec![num_experts, hidden_size, moe_intermediate_size],
                format!("{}.experts.down_proj.weight", scope_name),
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
        println!("Entering SparseMoeBlock forward: {}", tensor_name);
        println!("gate weight shape: {:?}", self.gate_weight.shape);
        // gate_output [sequence_chunk_size, batch_size, num_experts]
        let gate_output = hidden_states.matmul(
            &self.gate_weight,
            MatMulParams {
                a_row_step_macro: 6,
                b_row_step_macro: 128,
                column_step_macro: 16,
                a_row_step_micro: 3,
                b_row_step_micro: 128,
            },
            hidden_states.shape[0],
            format!("{}.gate", self.scope_name),
        );

        println!(
            "After gate matmul in SparseMoeBlock forward: {}",
            tensor_name
        );
        let (experts_indicator, indice_ptr, weight_ptr, topk_indices_ptr) = gate_output
            .experts_softmax_norm(
                self.num_experts,
                self.num_topk,
                format!("{}.router_probs", self.scope_name),
            );

        println!(
            "After experts_softmax_norm in SparseMoeBlock forward: {}",
            tensor_name
        );
        // nonlinear_product [num_experts, sequence_chunk_size, batch_size, intermediate_size]
        let nonlinear_product = hidden_states.experts_matmul_silu_mul_matmul(
            &self.experts_gate_weight,
            &self.experts_up_weight,
            experts_indicator,
            indice_ptr,
            MatMulParams {
                a_row_step_macro: 6,
                b_row_step_macro: 128,
                column_step_macro: 16,
                a_row_step_micro: 3,
                b_row_step_micro: 128,
            },
            format!("{}.gate_up", self.scope_name),
        );

        println!(
            "After experts_matmul_silu_mul_matmul in SparseMoeBlock forward: {}",
            tensor_name
        );
        // down_product [sequence_chunk_size, batch_size, num_experts_per_token, hidden_size]
        let down_product = nonlinear_product.experts_matmul_mul(
            &self.experts_down_weight,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            topk_indices_ptr,
            self.num_topk,
            MatMulParams {
                a_row_step_macro: 6,
                b_row_step_macro: 128,
                column_step_macro: 16,
                a_row_step_micro: 3,
                b_row_step_micro: 128,
            },
            format!("{}.down", self.scope_name),
        );

        println!(
            "After experts_matmul_mul in SparseMoeBlock forward: {}",
            tensor_name
        );

        let merge_tensor = down_product.experts_merge_add(
            residual,
            experts_indicator,
            indice_ptr,
            self.num_experts,
            format!("{}.merge", self.scope_name),
        );
        merge_tensor
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    // use crate::memory::allocator::allocate_init;

    #[test]
    fn test_sparse_moe_block() {
        let sequence_chunk_size = 1;
        let batch_size = 24;
        // let head_size = 128;

        let hidden_size = 256;
        let intermediate_size = 4 * hidden_size;
        let num_experts = 128;
        let top_k = 8;
        let norm_topk_prob = true;

        let cache = Rc::new(RefCell::new(Cache::<f32>::new(
            std::collections::HashMap::new(),
        )));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let sparse_moe = SparseMoeBlock::<f32>::new(
            // position_window_size,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            norm_topk_prob,
            "model.layers.0",
            cache.clone(),
            operator_queue.clone(),
        );

        let shape = vec![sequence_chunk_size, batch_size, hidden_size];
        let input = Tensor::from_cache(
            shape.clone(),
            String::from("model.layers.0.input_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        let residual = Tensor::from_cache(
            shape.clone(),
            String::from("model.layers.0.residual_tensor"),
            cache.clone(),
            operator_queue.clone(),
        );

        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        for i in 0..residual.shape.iter().product() {
            unsafe {
                residual.data.add(i).write(1.0);
            }
        }

        let output_tensor = sparse_moe.forward(
            &input,
            &residual,
            String::from("model.layers.0.output_tensor"),
        );

        let thread_num: usize = num_cpus::get();
        for (index, operator) in output_tensor.operator_queue.borrow().iter().enumerate() {
            println!("operator {} in queue", index);
            for i in 0..thread_num {
                operator.run(0, 1, batch_size, thread_num, i);
            }
        }
    }
}
