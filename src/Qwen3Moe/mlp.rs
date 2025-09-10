use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};
use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use super::super::memory::cache::Cache;
use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
use crate::compiler::operator::Operator;


#[derive(Clone)]
pub struct MLP<T> {
    sequence_chunk_size: usize,
    head_size: usize,
    gate_proj: Linear<T>,
    up_proj: Linear<T>,
    down_proj: Linear<T>,
    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> MLP<T>
where
    T: Copy + Default + Sub<Output = T> + Neg<Output = T> + Exp + NegInfinity + Sigmoid<T> + Sqrt,
{
    pub fn new(
        sequence_chunk_size: usize,
        head_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        // multiple_of: usize,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {

        let scope_name = format!("{}.mlp", parent_scope_name);
        Self {
            sequence_chunk_size: sequence_chunk_size,
            head_size: head_size,
            gate_proj: Linear::new(
                hidden_size,
                intermediate_size,
                sequence_chunk_size,
                format!("{}.gate_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            up_proj: Linear::new(
                hidden_size,
                intermediate_size,
                sequence_chunk_size,
                format!("{}.up_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            down_proj: Linear::new(
                intermediate_size,
                hidden_size,
                sequence_chunk_size,
                format!("{}.down_proj", scope_name),
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
        cpu_num: usize,
    ) -> Tensor<T> {
        // println!("{:?} {:?}", self.gate_proj.weight.shape ,hidden_states.shape);
        let gate_product = self
            .gate_proj
            .forward(hidden_states, format!("{}.linear1", self.scope_name));
        // println!("{:?} {:?}", hidden_states.shape, self.gate_proj.weight.shape);
        let up_product = self
            .up_proj
            .forward(hidden_states, format!("{}.linear3", self.scope_name));
        // println!("{:?} {:?}", hidden_states.shape, self.down_proj.weight.shape);

        let view_gate_product = gate_product.view(vec![
            gate_product.shape[0],
            gate_product.shape[1],
            gate_product.shape[2] / self.head_size,
            self.head_size ,
        ]);
        let view_up_product = up_product.view(vec![
            up_product.shape[0],
            up_product.shape[1] / self.head_size,
            self.head_size,
        ]);

        let nonlinear = view_gate_product.silu_mul(
            &view_up_product,
            format!("{}.nonlinear", self.scope_name),
        );

        let view_nonlinear = nonlinear.view(up_product.shape.clone());
        let down_product = self.down_proj.forward(&view_nonlinear, tensor_name);
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
