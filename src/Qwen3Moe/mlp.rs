use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::memory::cache::Cache;
// use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
use crate::compiler::operator::Operator;

#[derive(Clone)]
pub struct MLP<T> {
    // sequence_chunk_size: usize,
    // head_size: usize,
    gate_weight: Tensor<T>,
    up_weight: Tensor<T>,
    down_weight: Tensor<T>,
    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> MLP<T>
where
    T: Copy + Default + Sub<Output = T> + Neg<Output = T> + Exp + NegInfinity + Sigmoid<T> + Sqrt,
{
    pub fn new(
        // sequence_chunk_size: usize,
        // head_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = format!("{}.mlp", parent_scope_name);
        Self {
            // sequence_chunk_size: sequence_chunk_size,
            // head_size: head_size,
            gate_weight: Tensor::zeros(
                vec![hidden_size, intermediate_size],
                format!("{}.gate_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            up_weight: Tensor::zeros(
                vec![hidden_size, intermediate_size],
                format!("{}.up_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),

            down_weight: Tensor::zeros(
                vec![intermediate_size, hidden_size],
                format!("{}.down_proj.weight", scope_name),
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

        let nonlinear_product = hidden_states.matmul_silu_mul_matmul(
            &self.gate_weight,
            &self.up_weight,
            hidden_states.shape[1],
            self.gate_weight.shape[0],
            self.gate_weight.shape[1],
            crate::ptensor::matmul::MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            format!("{}.nonlinear_part1", self.scope_name),
        );

        let down_product = nonlinear_product.matmul_add(
            &self.down_weight,
            residual,
            hidden_states.shape[1],
            self.gate_weight.shape[0],
            self.gate_weight.shape[1],
            crate::ptensor::matmul::MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            format!("{}.nonlinear_part2", self.scope_name),
        );

        // let down_product = self.down_proj.forward(&nonlinear_product, tensor_name);
        // println!("{:?} {:?}", nonlinear.shape, self.w2.weight.shape);
        down_product
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    use crate::memory::allocator::allocate_init;

    #[test]
    fn test_feedforward() {
        let position_window_size = 4;
        let batch_size = 32;
        let head_size = 128;

        let hidden_size = 8192;
        let hidden_dim = 4 * hidden_size;

        let position_index = 1;

        let multiple_of = 256;

        let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let feedforward = FeedForward::<f32>::new(
            hidden_size,
            hidden_dim,
            head_size,
            multiple_of,
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
