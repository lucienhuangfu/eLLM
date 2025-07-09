use std::cell::RefCell;
use std::rc::Rc;
use std::ops::{Add, Sub, Div, Mul, AddAssign, Neg};
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};
use crate::kernel::generic::sigmoid::Sigmoid;

use super::super::memory::cache::Cache;
use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
use crate::compiler::zip_map::silu_mul_zip::SiluZipMap;
use crate::compiler::operator::Operator;

#[derive( Clone)]
pub struct FeedForward<T> {
    head_size: usize,
    w1: Linear<T>,
    w2: Linear<T>,
    w3: Linear<T>,
    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> FeedForward<T> 
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
        dim: usize,
        hidden_dim: usize,
        head_size: usize,
        multiple_of: usize,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        // println!("dim {} hidden_dim {}", dim, hidden_dim);
        let hidden_dim1 = 2 * hidden_dim / 3;
        let hidden_dim2 = multiple_of * ((hidden_dim1 + multiple_of - 1) / multiple_of);
        let scope_name = format!("{}.mlp", parent_scope_name);
        FeedForward {
            head_size: head_size,
            w1: Linear::new(
                dim,
                hidden_dim2,
                1,
                format!("{}.up_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            w2: Linear::new(
                hidden_dim2,
                dim,
                1,
                format!("{}.down_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            w3: Linear::new(
                dim,
                hidden_dim2,
                1,
                format!("{}.gate_proj", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            scope_name: scope_name,
            cache: cache,
            operator_queue: operator_queue,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor<T>, tensor_name: String,cpu_num:usize) -> Tensor<T> {
        // println!("{:?} {:?}", self.w1.weight.shape ,hidden_states.shape);
        let linear1 = self
            .w1
            .forward(hidden_states, format!("{}.linear1", self.scope_name));
        // println!("{:?} {:?}", hidden_states.shape, self.w1.weight.shape);
        let linear3 = self
            .w3
            .forward(hidden_states, format!("{}.linear3", self.scope_name));
        // println!("{:?} {:?}", hidden_states.shape, self.w3.weight.shape);
        

        let view_linear1 = linear1.view(vec![linear1.shape[0], linear1.shape[1] /(self.head_size *2),self.head_size*2]);
        let view_linear3 = linear3.view(vec![linear3.shape[0], linear3.shape[1]/(self.head_size*2), self.head_size*2]);


        let nonlinear = view_linear1.zip_mapv(
            &view_linear3,
            Operator::SiluMulZipMap(SiluZipMap::new(self.head_size,view_linear1.shape[1], cpu_num)),
            false,
            format!("{}.nonlinear", self.scope_name),
        );

        let view_nonlinear = nonlinear.view(linear1.shape.clone());
        let linear2 = self.w2.forward(&view_nonlinear, tensor_name);
        // println!("{:?} {:?}", nonlinear.shape, self.w2.weight.shape);
        linear2
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    use crate::memory::allocator::allocate_init;

    #[test]
    fn test_feedforward() {
        let batch_size = 32;
        let position_index = 1;
        let hidden_size = 8192;
        let hidden_dim = 4 * hidden_size;
        let head_size = 128;
        let multiple_of = 256;


        let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let feedforward = FeedForward::<f32>::new( hidden_size, hidden_dim, head_size, multiple_of, "model.layers.0", cache.clone(), operator_queue.clone());

        let shape = vec![batch_size, hidden_size];
        let input = Tensor::from_cache(shape.clone(), String::from("model.layers.0.input_tensor"), cache.clone(), operator_queue.clone());
        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let output_tensor = feedforward.forward(&input, String::from("model.layers.0.output_tensor"), num_cpus::get());

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

    #[test]
    fn test_feedforward_f16() {
        let batch_size = 32;
        let position_index = 1;
        let hidden_size = 8192;
        let head_size = 128;
        let hidden_dim = 4 * hidden_size;
        let multiple_of = 256;


        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
        let operator_queue = Rc::new(RefCell::new(Vec::new()));

        let feedforward = FeedForward::<f16>::new(hidden_size, hidden_dim, head_size, multiple_of, "model.layers.0", cache.clone(), operator_queue.clone());

        let shape = vec![batch_size, hidden_size];
        let input = Tensor::from_cache(shape.clone(), String::from("model.layers.0.input_tensor"), cache.clone(), operator_queue.clone());
        for i in 0..input.shape.iter().product() {
            unsafe {
                input.data.add(i).write(1.0);
            }
        }

        let output_tensor = feedforward.forward(&input, String::from("model.layers.0.output_tensor"), num_cpus::get());

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


}
