use std::ops::{AddAssign, Neg, Sub};
use std::rc::Rc;

use crate::common::num_traits::Sqrt;
use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};

use super::super::common::matmul_params::MatMulParams;
use super::super::runtime::tensor::{Tensor, TensorCtx};
use super::names::DenseMlpTensorNames;

#[derive(Clone)]
pub struct MLP<T>
where
    T: Copy + PartialOrd,
{
    // sequence_chunk_size: usize,
    // head_size: usize,
    gate_weight: Tensor<T>,
    up_weight: Tensor<T>,
    down_weight: Tensor<T>,
    scope_name: String,
    ctx: Rc<TensorCtx<T>>,
}

impl<T> MLP<T>
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
        + AddAssign,
{
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        names: DenseMlpTensorNames,
        ctx: Rc<TensorCtx<T>>,
    ) -> Self {
        Self {
            gate_weight: ctx.zeros(
                vec![hidden_size, intermediate_size],
                names.gate_proj,
            ),
            up_weight: ctx.zeros(
                vec![hidden_size, intermediate_size],
                names.up_proj,
            ),

            down_weight: ctx.zeros(
                vec![intermediate_size, hidden_size],
                names.down_proj,
            ),
            scope_name: names.scope,
            ctx: ctx,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        _tensor_name: String,
    ) -> Tensor<T> {
        let gate_product = hidden_states.matmul(
            &self.gate_weight,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            hidden_states.shape[0],
            false,
            format!("{}.gate", self.scope_name),
        );

        let up_product = hidden_states.matmul(
            &self.up_weight,
            MatMulParams {
                a_row_step_macro: 16,
                b_row_step_macro: 16,
                column_step_macro: 16,
                a_row_step_micro: 8,
                b_row_step_micro: 8,
            },
            hidden_states.shape[0],
            false,
            format!("{}.up", self.scope_name),
        );

        let nonlinear_product = gate_product.add(
            &up_product,
            format!("{}.nonlinear_part1", self.scope_name),
        );

        let down_product = nonlinear_product.matmul_add(
            &self.down_weight,
            residual,
            MatMulParams {
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

    use crate::mem_mgr::allocator::allocate_init;

    /*
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
                operator.run(1, 0, thread_num, i, &[], &[], &mut Vec::new());
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
                operator.run(1, 0, thread_num, i, &[], &[], &mut Vec::new());
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






