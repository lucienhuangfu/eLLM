use std::cell::RefCell;
// use std::iter::zip;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::memory::cache::Cache;
use super::super::memory::allocator::allocate_init;
use super::tensor_utils::get_strides;
use crate::init::matmul_params::matmulParams;

use super::super::compiler::map::experts_softmax_norm::ExpertsSoftmaxNorm;
use super::super::compiler::map::lookup_rms_map::LookupRMSMap;
use super::super::compiler::map::rms_map::RMSMap;
use super::super::compiler::map::topk_softmax::TopKSoftmax;
// use super::super::compiler::mul::attention_mul_add::AttentionMul;
use super::super::compiler::mul::attention_add::AttentionAdd;
use super::super::compiler::mul::experts_merge_add::ExpertsmatmulMergeAdd;
use super::super::compiler::mul::experts_matmul_mul::ExpertsMatmulMul;
use super::super::compiler::mul::experts_matmul_silu_mul_matmul::ExpertsMatmulSilu;
use super::super::compiler::mul::matmul::matmul;
use super::super::compiler::mul::matmul3::matmul3;
use super::super::compiler::mul::matmul_add::matmulAdd;
use super::super::compiler::mul::matmul_silu_mul_matmul::matmulSilu;
use super::super::compiler::mul::matmul_topk::matmulTopK;
use super::super::compiler::operator::Operator;
use super::super::compiler::zip_map::add_zip::AddZipMap;
use super::super::compiler::zip_map::complex_zip::ComplexZipMap;
use super::super::compiler::zip_map::silu_mul_zip::SiluMulZipMap;
use crate::compiler::zip_map::add_rms_zip::AddRMSZipMap;
use super::super::compiler::mul::experts_routing::ExpertsRouting;

#[derive(Clone)]
pub struct Tensor<T> {
    pub data: *mut T,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub tensor_name: String,
    pub cache: Rc<RefCell<Cache<T>>>,
    pub operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    // pub size: usize,
    // pub is_contiguous: bool,
}

impl<T> Tensor<T>
where
    T: Copy + Default + Sub<Output = T> + Neg<Output = T> + Exp + NegInfinity + Sigmoid<T> + Sqrt,
{
    pub fn add(&self, b_tensor: &Tensor<T>, tensor_name: String) -> Self {
        let output_tensor = <Tensor<T>>::from_cache(
            self.shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );
        let operator = Operator::AddZipMap(AddZipMap::new(
            self.data,
            b_tensor.data,
            output_tensor.data,
            self.shape[1],
            self.shape[2],
            self.shape[3],
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }


    pub fn add_rms(
        &self,
        b_tensor: &Tensor<T>,
        weight: *const T,
        eps: T,
        tensor_name: String,
    ) -> Self {
        let output_tensor = Tensor::from_cache(
            self.shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::AddRMSZipMap(AddRMSZipMap::new(
            self.data,
            b_tensor.data,
            output_tensor.data,
            self.shape[1],
            self.shape[2],
            // weight,
            eps,
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn attention_add(
        &self,
        k_tensor: &Tensor<T>,
        v_tensor: &Tensor<T>,
        residual: &Tensor<T>,
        inverse_sqrt_head: T,
        scope_name: String,
    ) -> Self {
        let output_shape = self.shape.clone();
        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::AttentionAdd(AttentionAdd::new(
            self.data,
            k_tensor.data,
            v_tensor.data,
            residual.data,
            output_tensor.data,
            self.shape[1],
            self.shape[2],
            k_tensor.shape[2],
            self.shape[3],
            k_tensor.strides.clone(),
            inverse_sqrt_head,
        ));

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn complex_mul(
        &self,
        b_tensor: &Tensor<T>,
        sequence_length: usize,
        tensor_name: String,
    ) -> Self {
        let output_to_kv = if self.shape[0] == sequence_length {
            false
        } else {
            true
        };
        let output_shape = vec![sequence_length, self.shape[1], b_tensor.shape[2]];
        let output_tensor = Tensor::from_cache(
            output_shape,
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );
        let operator = Operator::ComplexZipMap(ComplexZipMap::new(
            self.data,
            b_tensor.data,
            output_tensor.data,
            // sequence_length,
            self.shape[1],
            self.shape[2],
            self.shape[3],
            output_to_kv,
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn experts_merge_add(
        &self,
        residual: &Tensor<T>,
        scope_name: String,
    ) -> Self {
        // down_weights [num_experts, hidden_size, intermediate_size]
        // output [sequence_chunk_size, batch_size, hidden_size]
        let output_shape = vec![self.shape[0], self.shape[1], down_weights.shape[1]];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::ExpertsMergeAdd(ExpertsMergeAdd::new(
            self.data,
            output_tensor.data,
            self.shape[1],
            down_weights.shape[1],
            self.shape[2],
        ));

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn experts_matmul_mul(
        &self,
        down_weights: &Tensor<T>,
        experts_routing: ExpertRouting<T>,
        params: matmulParams,
        scope_name: String,
    ) -> Self {
        // down_weights [num_experts, hidden_size, intermediate_size]
        // output [sequence_chunk_size, batch_size, hidden_size]
        let output_shape = vec![self.shape[0], self.shape[1], down_weights.shape[1]];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::ExpertsMatmulMul(ExpertsMatmulMul::new(
            self.data,
            down_weights.data,
            output_tensor.data,
            experts_routing,
            self.shape[1],
            down_weights.shape[1],
            self.shape[2],
            params.a_row_step_macro,
            params.b_row_step_macro,
            params.column_step_macro,
            params.a_row_step_micro,
            params.b_row_step_micro,
        ));

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn experts_matmul_silu_mul_matmul(
        &self,
        gate_weights: &Tensor<T>,
        up_weights: &Tensor<T>,
        experts_routing: ExpertRouting<T>,
        params: matmulParams,
        tensor_name: String,
    ) -> Self {
        // gate_weights [num_experts, intermediate_size, hidden_size]
        // output [sequence_chunk_size, batch_size, intermediate_size]
        let output_shape = vec![self.shape[0], self.shape[1], gate_weights.shape[1]];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::ExpertsMatmulSiluMulMatmul(ExpertsMatmulSilu::new(
            self.data,
            gate_weights.data,
            up_weights.data,
            experts_routing,
            output_tensor.data,
            self.shape[1],
            gate_weights.shape[1],
            self.shape[2],
            params.a_row_step_macro,
            params.b_row_step_macro,
            params.column_step_macro,
            params.a_row_step_micro,
            params.b_row_step_micro,
        ));

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn experts_softmax_norm(
        &self,
        num_experts: usize,
        num_experts_per_tok: usize,
        scope_name: String,
    ) -> ExpertsRouting<T> {
        // Create ExpertsRouting with the corrected parameters
        let experts_routing = ExpertsRouting::new(
            self.shape[0], // sequence_chunk_size
            self.shape[1], // batch_size
            num_experts,
            num_experts_per_tok,
            &mut self.cache.borrow_mut(),
        );

        let operator = Operator::ExpertsSoftmaxNorm(ExpertsSoftmaxNorm::new(
            self.data,
            experts_routing,
            self.shape[1],
            num_experts,
            num_experts_per_tok,
        ));
        self.operator_queue.borrow_mut().push(operator);
        experts_routing
    }

    pub fn from_cache(
        shape: Vec<usize>,
        tensor_name: String,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let length: usize = shape.iter().product();
        let data = cache.borrow_mut().get(&tensor_name, length);
        let strides = get_strides(&shape);
        Tensor {
            data: data,
            shape: shape.clone(),
            strides: strides,
            tensor_name: tensor_name,
            cache: cache.clone(),
            operator_queue: operator_queue.clone(),
        }
    }

    pub fn matmul(
        &self,
        tensor2: &Tensor<T>,
        params: matmulParams,
        sequence_length: usize,
        tensor_name: String,
    ) -> Self {
        let output_shape = vec![self.shape[0], self.shape[1], tensor2.shape[0]];

        let output_to_kv = if self.shape[0] <= sequence_length {
            false
        } else {
            true
        };
        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = unsafe {
            Operator::matmul(matmul::new(
                self.data,
                tensor2.data,
                output_tensor.data,
                output_to_kv,
                params,
                self.shape[1],
                tensor2.shape[0],
                self.shape[2],
            ))
        };

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn matmul_add(
        &self,
        tensor2: &Tensor<T>,
        tensor3: &Tensor<T>,
        params: matmulParams,
        tensor_name: String,
    ) -> Self {
        let output_shape = vec![self.shape[0], self.shape[1], tensor2.shape[0]];

        let a_row = self.shape[1];
        let b_row = tensor2.shape[0];
        let column = self.shape[2];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::matmulAdd(matmulAdd::new(
            self.data,
            tensor2.data,
            tensor3.data,
            output_tensor.data,
            a_row,
            b_row,
            column,
            params.a_row_step_macro,
            params.b_row_step_macro,
            params.column_step_macro,
            params.a_row_step_micro,
            params.b_row_step_micro,
        ));

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn matmul3(
        &self,
        q_weight: &Tensor<T>,
        k_weight: &Tensor<T>,
        v_weight: &Tensor<T>,
        position_embedding: &Tensor<T>,
        head_dim: usize,
        params: matmulParams,
        scope_name: String,
    ) -> (Self, Self, Self) {
        // let head_dim = 128; // Fixed head dimension
        let a_h_row = self.shape[1];
        let col = self.shape[2];
        // let b_row = q_weight.shape[1];

        let q_state = Tensor::from_cache(
            vec![self.shape[0], self.shape[1], q_weight.shape[0]],
            format!("{}.q_state", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let k_state = Tensor::from_cache(
            vec![self.shape[0], self.shape[1], k_weight.shape[0]],
            format!("{}.k_state", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let v_state = Tensor::from_cache(
            vec![self.shape[0], self.shape[1], v_weight.shape[0]],
            format!("{}.v_state", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::matmul3(matmul3::new(
            self.data,
            q_weight.data,
            q_state.data,
            k_weight.data,
            k_state.data,
            v_weight.data,
            v_state.data,
            position_embedding.data,
            head_dim,
            a_h_row,
            col,
            q_weight.shape[0],
            k_weight.shape[0],
            params.a_row_step_macro,
            params.b_row_step_macro,
            params.column_step_macro,
            params.a_row_step_micro,
            params.b_row_step_micro,
        ));

        self.operator_queue.borrow_mut().push(operator);
        (q_state, k_state, v_state)
    }

    pub fn matmul_silu_mul_matmul(
        &self,
        tensor2: &Tensor<T>,
        tensor3: &Tensor<T>,
        params: matmulParams,
        tensor_name: String,
    ) -> Self {
        // hidden_tensor [sequence_chunk_size, batch_size, hidden_size]
        // tensor2 [intermediate_size, hidden_size]
        // output [sequence_chunk_size, batch_size, intermediate_size]

        let a_row = self.shape[1];
        let b_row = tensor2.shape[0];
        let column = self.shape[2];

        let output_shape = vec![self.shape[0], a_row, b_row];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::matmulSiluMulmatmul(matmulSilu::new(
            self.data,
            tensor2.data,
            tensor3.data,
            output_tensor.data,
            a_row,
            b_row,
            column,
            params.a_row_step_macro,
            params.b_row_step_macro,
            params.column_step_macro,
            params.a_row_step_micro,
            params.b_row_step_micro,
        ));

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn matmul_topk(
        &self,
        tensor2: &Tensor<T>,
        params: matmulParams,
        thread_num: usize,
        scope_name: String,
    ) -> (*const usize, Self, Self) {
        let a_row = self.shape[1];
        let b_row = tensor2.shape[1];
        let column = self.shape[2];

        let output_shape = vec![self.shape[0], self.shape[1], tensor2.shape[0]];

        /*
        let indice_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.indices", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        ); */

        let indice_ptr = allocate_init(output_shape.iter().product(), 0usize);
        //  = self.cache.borrow_mut().get(&format!("{}.indices", scope_name), output_shape.iter().product());

        let value_tensor = Tensor::<T>::from_cache(
            output_shape.clone(),
            format!("{}.values", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let sum_tensor = Tensor::from_cache(
            vec![self.shape[0], self.shape[1], thread_num],
            format!("{}.sums", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::matmulTopK(matmulTopK::new(
            self.data,
            tensor2.data,
            indice_ptr,
            value_tensor.data,
            sum_tensor.data,
            a_row,
            b_row,
            column,
            params.a_row_step_macro,
            params.b_row_step_macro,
            params.column_step_macro,
            params.a_row_step_micro,
            params.b_row_step_micro,
        ));

        self.operator_queue.borrow_mut().push(operator);
        (indice_ptr, value_tensor, sum_tensor)
    }

    pub fn permute(&self, dims: Vec<usize>) -> Self {
        let shape: Vec<usize> = dims
            .clone()
            .into_iter()
            .map(|index| self.shape[index])
            .collect();
        let strides: Vec<usize> = dims
            .clone()
            .into_iter()
            .map(|index| self.strides[index])
            .collect();
        // let maximum_shape: Vec<usize> = dims.clone().into_iter().map(|index| self.maximum_shape[index]).collect();
        // let tensor = self._view(shape, strides, tensor_name);
        // tensor.set_contiguous(false) ;
        Tensor {
            data: self.data,
            shape: shape,
            strides: strides,
            tensor_name: self.tensor_name.clone(),
            cache: self.cache.clone(),
            operator_queue: self.operator_queue.clone(),
        }
    }

    pub fn rms(&self, eps: T, scope_name: String) -> Self {
        let output_tensor = Tensor::<T>::from_cache(
            self.shape.clone(),
            format!("{}.rms_output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::RMSMap(RMSMap::new(
            self.data,
            output_tensor.data,
            self.shape[1],
            self.shape[2],
            eps,
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn silu_mul(&self, b_tensor: &Tensor<T>, tensor_name: String) -> Self {
        let output_tensor = Tensor::<T>::from_cache(
            self.shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );
        let operator = Operator::SiluMulZipMap(SiluMulZipMap::new(
            self.data,
            b_tensor.data,
            output_tensor.data,
            self.shape[1],
            self.shape[2],
            self.shape[3],
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn topk_softmax(
        &self,
        indices_ptr: *const usize,
        // values_tensor: &Tensor<T>,
        sums_tensor: &Tensor<T>,
        topk_size: usize,
        scope_name: String,
        // value_tensor_name: String,
    ) -> (*const usize, Self) {
        /* 
        let indice_tensor = Tensor::from_cache(
            vec![self.shape[0], self.shape[1], topk_size],
            format!("{}.output_indice", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );*/

        let output_shape = vec![self.shape[0], self.shape[1], topk_size];
        let indice_ptr = allocate_init(output_shape.iter().product(), 0usize);

        let value_tensor = Tensor::from_cache(
            vec![self.shape[0], self.shape[1], topk_size],
            format!("{}.output_value", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        

        let operator = Operator::TopKSoftmax(TopKSoftmax::new(
            //indices_tensor.data,
            indices_ptr,
            self.data,
            // values_tensor.data,
            sums_tensor.data,
            indice_ptr,
            value_tensor.data,
            self.shape[1],
            topk_size,
        ));

        self.operator_queue.borrow_mut().push(operator);
        (indice_ptr, value_tensor)
    }

    pub fn transpose(&mut self, index1: usize, index2: usize) -> Self {
        let mut dims: Vec<usize> = (0..self.shape.len()).collect();
        dims.swap(index1, index2);
        // self.set_contiguous(false);
        self.permute(dims)
    }

    pub fn view(&self, shape: Vec<usize>) -> Self {
        let strides = get_strides(&shape);
        self._view(shape, strides)
    }

    pub fn zeros(
        shape: Vec<usize>,
        tensor_name: String,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let length: usize = shape.iter().product();
        let v = Self::from_cache(shape, tensor_name, cache, operator_queue);
        (0..length).for_each(|x| unsafe {
            *v.data.add(x) = T::default();
        });
        v
    }

    fn _view(&self, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        // let size: usize = shape.iter().product();
        Tensor {
            data: self.data,
            shape: shape.clone(),
            strides: strides,
            tensor_name: self.tensor_name.clone(),
            cache: self.cache.clone(),
            operator_queue: self.operator_queue.clone(),
        }
    }

    pub fn lookup_rms(
        input_sequences: *mut usize,
        word_embedding: &Tensor<T>,
        batch_size: usize,
        eps: T,
        sequence_chunk_size: usize,
        scope_name: String,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> (Self, Self) {
        let output_hidden_tensor = Tensor::from_cache(
            vec![
                sequence_chunk_size,
                batch_size,
                word_embedding.shape[1],
            ],
            format!("{}.output_hidden", scope_name),
            cache.clone(),
            operator_queue.clone(),
        );

        let output_normal_tensor = Tensor::from_cache(
            vec![
                sequence_chunk_size,
                batch_size,
                word_embedding.shape[1],
            ],
            format!("{}.output_normal", scope_name),
            cache.clone(),
            operator_queue.clone(),
        );

        let operator = Operator::LookupRMSMap(LookupRMSMap::new(
            input_sequences,
            word_embedding.data,
            output_hidden_tensor.data,
            output_normal_tensor.data,
            batch_size,
            word_embedding.shape[1],
            eps,
        ));

        operator_queue.borrow_mut().push(operator);
        (output_hidden_tensor, output_normal_tensor)
    }

    /*
    pub fn reduce(
        &self,
        sequences: *mut usize,
        sequence_length: usize,
        mut operator: Operator<T>,
        tensor_name: String,
        // cache: &mut Cache<T>,
        // operator_queue: &mut Vec<Operator<T>>,
    ) {
        /*
        let output_shape = vec![sequence_length, self.shape[0]];
        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        ); */

        let chunks = chunk_reduce(
            self.shape.clone(),
            self.data,
            self.strides.clone(),
            sequences,
            vec![1],
        );
        let mut extended_chunks = vec![];
        for step in (0..self.shape[0] * sequence_length).step_by(self.shape[0]) {
            for (a_ptr, mut b_ptr) in chunks.iter().cloned() {
                unsafe {
                    b_ptr.ptr = b_ptr.ptr.add(step);
                    extended_chunks.push((a_ptr, b_ptr));
                }
            }
        }
        operator.set_reduce_chunk(extended_chunks);
        self.operator_queue.borrow_mut().push(operator);
        // output_tensor
    }
    pub fn mapv(
        &self,
        mut operator: Operator<T>,
        tensor_name: String,
        // cache: &mut Cache<T>,
        // operator_queue: &mut Vec<Operator<T>>,
    ) -> Self {
        let output_tensor = Tensor::from_cache(
            self.shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );
        let chunks = chunk_map(
            self.shape.clone(),
            self.strides.clone(),
            self.data,
            output_tensor.data,
        );
        operator.set_map_chunk(chunks);
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn zip_mapv(
        &self,
        b_tensor: &Tensor<T>,
        mut operator: Operator<T>,
        partial_broadcast: bool,
        tensor_name: String,
        // cache: &mut Cache<T>,
        // operator_queue: &mut Vec<Operator<T>>,
    ) -> Self {
        let broadcast_shape = get_broadcast_shape(&self.shape, &b_tensor.shape);
        let a_strides = get_aligned_strides(&self.shape, &broadcast_shape);
        let b_strides = get_aligned_strides(&b_tensor.shape, &broadcast_shape);

        let (output_shape, output_strides) = if partial_broadcast == true {
            let offset = broadcast_shape.len() - self.shape.len();
            let mut output_strides: Vec<usize> = vec![0; offset];
            output_strides.extend(self.strides.iter().cloned());

            (self.shape.clone(), output_strides)
        } else {
            (broadcast_shape.clone(), get_strides(&broadcast_shape))
        };

        let output_tensor = Tensor::from_cache(
            output_shape,
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let chunks = chunk_zipmap(
            broadcast_shape,
            self.data,
            a_strides,
            b_tensor.data,
            b_strides,
            output_tensor.data,
            output_strides,
        );
        operator.set_zipmap_chunk(chunks);
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }


    }*/
}

unsafe impl<T: Copy + Default + Send + Sync> Send for Tensor<T> {}
unsafe impl<T: Copy + Default + Send + Sync> Sync for Tensor<T> {}

#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    // use num_cpus;
    // use std::sync::Arc;
    // use std::sync::Barrier;
    use std::collections::HashMap;
    use std::thread;
    use super::*;

    
}
