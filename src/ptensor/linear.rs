use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Barrier};

use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg};
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;

use super::super::memory::cache::Cache;
use super::super::ptensor::tensor::Tensor;
use crate::compiler::mul::mat_mul::MatMul;
use crate::init::matmul_params::MatMulParams;
use crate::compiler::operator::Operator;

#[derive(Clone)]
pub struct Linear<T> {
    pub weight: Tensor<T>,
    // pub bias: Tensor,
    sequence_length: usize,
    scope_name: String,
    pub cache: Rc<RefCell<Cache<T>>>,
    pub operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> Linear<T> 
where T: Copy 
    + Default 
    + Sub<Output = T>
    + Neg<Output = T>
    + Exp
    + NegInfinity
    + Sigmoid<T>
    + Sqrt
{
    pub fn new(
        in_features: usize,
        out_features: usize,
        sequence_length: usize,
        scope_name: String,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let weight = Tensor::zeros(
            vec![out_features, in_features],
            format!("{}.weight", scope_name),
            cache.clone(),
            operator_queue.clone(),
        );
        // let bias = Tensor::zeros(vec![out_features], format!("{}.bias", scope_name), true);
        Linear {
            weight: weight,
            // bias: bias,
            sequence_length: sequence_length,
            scope_name: scope_name,
            cache: cache,
            operator_queue: operator_queue,
        }
    }

    pub fn forward(&self, input: &Tensor<T>, tensor_name: String) -> Tensor<T> {
        //[position_window_size, batch_size , hidden_size]   <- [position_window_size, batch_size, hidden_size]   [ hidden_size, hidden_size]
        let a_row = input.shape[1];
        let b_row =  self.weight.shape[0];
        let column = self.weight.shape[1];
        let a_row_step_macro = 16;
        let b_row_step_macro = 16;
        let column_step_macro = 16;
        let a_row_step_micro = 8;
        let b_row_step_micro = 8;

        let params: MatMulParams = MatMulParams {
            a_row,
            b_row,
            column,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
        };

        let thread_num: usize = num_cpus::get();
        // let barrier = Barrier::new(thread_num);
        // let barrier_arc = Arc::new(barrier);
        let runner = Operator::MatMul(MatMul::new(
            a_row,
            b_row,
            column,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
            self.sequence_length,
            //chunks,
            thread_num,
            barrier_arc,
        ));

        let product = input.matmul(&self.weight, runner, params, self.sequence_length, tensor_name);
        product
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    // use std::f16;
    use std::thread;

    #[test]
    fn test_linear_batch_size_1() {
        let sequence_length = 1;
        let position_window_size = 4;
        let batch_size = 32;
        let head_num = 64;
        let head_size = 128;
        
        let hidden_size = head_num * head_size;
        


        let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let linear = Linear::<f32>::new(hidden_size, hidden_size, sequence_length, String::from("model.layers.0"), cache.clone(), operator_queue.clone());
        
        for i in 0..linear.weight.shape.iter().product() {
            unsafe { linear.weight.data.add(i).write(1.0f32) };
        }
        
        let shape1 = vec![position_window_size, batch_size, hidden_size];

        let input = Tensor::from_cache(shape1, String::from("model.layer.0.input_tensor"), cache.clone(), operator_queue.clone());
        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let output_shape = vec![position_window_size, batch_size, hidden_size];
        let size3 = output_shape.iter().product();
        let mut result = vec![0.0; size3];
        for i in 0..hidden_size {
            result[i] = hidden_size as f32;
        }

        let output_tensor = linear.forward(&input, String::from("model.layer.0.self_attn.value_tensor"));
        
        let thread_num: usize = num_cpus::get();
        for i in 0..thread_num {
            output_tensor.operator_queue.borrow()[0].run(1, 0, i);
        }

        let output_slice = unsafe { std::slice::from_raw_parts(output_tensor.data, size3) };
        assert_relative_eq!(output_slice, &result[..], max_relative = 1e-6);
    }
    
}
