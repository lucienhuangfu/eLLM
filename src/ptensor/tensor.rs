use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::init::tensor_utils::get_strides;
use super::super::memory::allocator::allocate_init;
use super::super::memory::cache::Cache;
use crate::init::matmul_params::MatMulParams;

use super::super::compiler::map::experts_softmax_norm::ExpertsSoftmaxNorm;
use super::super::compiler::map::lookup_rms_map::LookupRMSMap;
use super::super::compiler::map::rms_map::RMSMap;
use super::super::compiler::map::topk_softmax::TopKSoftmax;
use super::super::compiler::mul::attention::Attention;
use super::super::compiler::mul::experts_matmul_mul::ExpertsMatMulMul;
use super::super::compiler::mul::experts_matmul_silu_mul_matmul::ExpertsMatMulSilu;
use super::super::compiler::mul::experts_merge_add::ExpertsMergeAdd;
use super::super::compiler::mul::matmul::MatMul;
use super::super::compiler::mul::matmul3::MatMul3;
use super::super::compiler::mul::matmul_add::MatMulAdd;
use super::super::compiler::mul::matmul_silu_mul_matmul::MatMulSilu;
use super::super::compiler::mul::matmul_topk::MatMulTopK;
use super::super::compiler::operator::Operator;
use super::super::compiler::zip_map::add_zip::AddZipMap;
use super::super::compiler::zip_map::complex_zip::ComplexZipMap;
use super::super::compiler::zip_map::silu_mul_zip::SiluMulZipMap;
use crate::compiler::zip_map::add_rms_zip::AddRMSZipMap;

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

    pub fn attention(
        &self,
        k_tensor: &Tensor<T>,
        v_tensor: &Tensor<T>,
        // residual: &Tensor<T>,
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

        let operator = Operator::Attention(Attention::new(
            self.data,
            k_tensor.data,
            v_tensor.data,
            // residual.data,
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
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        num_experts: usize,
        scope_name: String,
    ) -> Self {
        // output [sequence_chunk_size, batch_size, hidden_size]
        let output_shape = vec![self.shape[0], self.shape[1], self.shape[3]];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::ExpertsMergeAdd(ExpertsMergeAdd::new(
            self.data,
            residual.data,
            experts_indicator,
            indice_ptr,
            output_tensor.data,
            self.shape[0],
            self.shape[1],
            num_experts,
            self.shape[2],
            self.shape[3],
        ));

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn experts_matmul_mul(
        &self,
        down_weights: &Tensor<T>,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        weight_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        // sequence_chunk_size: usize,
        // batch_size: usize,
        num_experts_per_tok: usize,
        params: MatMulParams,
        scope_name: String,
    ) -> Self {
        // down_weights [num_experts, hidden_size, intermediate_size]
        // output [sequence_chunk_size, batch_size, num_experts_per_token, hidden_size]
        let output_shape = vec![
            self.shape[1],
            self.shape[2],
            num_experts_per_tok,
            down_weights.shape[1],
        ];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::ExpertsMatMulMul(ExpertsMatMulMul::new(
            self.data,
            down_weights.data,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            topk_indices_ptr,
            output_tensor.data,
            self.shape[2],
            down_weights.shape[1],
            down_weights.shape[0],
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
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        // weight_ptr: *mut T,
        params: MatMulParams,
        scope_name: String,
    ) -> Self {
        // gate_weights [num_experts, intermediate_size, hidden_size]
        // output [num_experts, sequence_chunk_size , batch_size, intermediate_size]
        let output_shape = vec![
            gate_weights.shape[0],
            self.shape[0],
            self.shape[1],
            gate_weights.shape[1],
        ];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::ExpertsMatMulSiluMulMatMul(ExpertsMatMulSilu::new(
            self.data,
            gate_weights.data,
            up_weights.data,
            expertsIndicator,
            indice_ptr,
            output_tensor.data,
            self.shape[1],
            gate_weights.shape[1],
            gate_weights.shape[0],
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
    ) -> (*mut bool, *mut bool, *mut T, *mut usize) {
        // [(experts_id, [(token_id, weight)])]
        // sorted_ids: Vec<(usize, Vec<(usize, T)>)>,

        // [expert_num] bool
        let experts_indicator = unsafe { allocate_init(num_experts, false) };
        // [expert_num, sequence_chunk_size * batch_size] indice bool vec<bool>
        // [expert_num, sequence_chunk_size * batch_size] weight f16
        let length = num_experts * self.shape[0] * self.shape[1];
        let indice_ptr = unsafe { allocate_init(length, false) };
        let weight_ptr = unsafe { allocate_init(length, T::default()) };
        let mut topk_indices_ptr =
            unsafe { allocate_init(num_experts_per_tok * self.shape[0] * self.shape[1], 0usize) };
        // vec![0usize; num_experts * self.shape[0] * self.shape[1]];

        let operator = Operator::ExpertsSoftmaxNorm(ExpertsSoftmaxNorm::new(
            self.data,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            topk_indices_ptr,
            self.shape[0],
            self.shape[1],
            num_experts,
            num_experts_per_tok,
        ));
        self.operator_queue.borrow_mut().push(operator);
        (experts_indicator, indice_ptr, weight_ptr, topk_indices_ptr)
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
        params: MatMulParams,
        sequence_length: usize,
        scope_name: String,
    ) -> Self {
        let output_shape = vec![self.shape[0], self.shape[1], tensor2.shape[0]];

        let output_to_kv = if self.shape[0] <= sequence_length {
            false
        } else {
            true
        };
        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );
        tensor2.data;
        println!(
            "Before matmul operator creation in Tensor matmul: {}",
            scope_name
        );

        let operator = unsafe {
            Operator::MatMul(MatMul::new(
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
        println!("After matmul in Tensor matmul: {}", scope_name);

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn matmul_add(
        &self,
        tensor2: &Tensor<T>,
        tensor3: &Tensor<T>,
        params: MatMulParams,
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

        let operator = Operator::MatMulAdd(MatMulAdd::new(
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
        params: MatMulParams,
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

        let operator = Operator::MatMul3(MatMul3::new(
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
        params: MatMulParams,
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

        let operator = Operator::MatMulSiluMulMatMul(MatMulSilu::new(
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

    pub fn matmul_local_topk(
        &self,
        tensor2: &Tensor<T>,
        params: MatMulParams,
        thread_num: usize,
        topk: usize,
        scope_name: String,
    ) -> (*const usize, Self) {
        let m = self.shape[0] * self.shape[1];
        let k = self.shape[2];
        let n = tensor2.shape[0];

        // MatMulTopK uses internal thread detection for buffer layout
        // let thread_max = MatMulTopK::<T>::detect_threads();
        let output_shape = vec![self.shape[0], self.shape[1], thread_num * topk];

        let indice_ptr = unsafe { allocate_init(output_shape.iter().product(), 0usize) };

        let value_tensor = Tensor::<T>::from_cache(
            output_shape.clone(),
            format!("{}.values.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = unsafe {
            Operator::MatMulTopK(MatMulTopK::new(
                self.data,
                tensor2.data,
                indice_ptr,
                value_tensor.data,
                m, // a_row
                n, // b_row
                k, // column
                params.a_row_step_macro,
                params.b_row_step_macro,
                params.column_step_macro,
                params.a_row_step_micro,
                params.b_row_step_micro,
                m, // batch_max
                topk,
            ))
        };

        self.operator_queue.borrow_mut().push(operator);
        (indice_ptr, value_tensor)
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
            format!("{}.output", scope_name),
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
        sums_tensor: &Tensor<T>,
        output_sequences: *mut usize,
        num_topk: usize,
        scope_name: String,
    ) -> (*const usize, Self) {
        let output_shape = vec![self.shape[0], self.shape[1], num_topk];
        let indice_ptr = allocate_init(output_shape.iter().product(), 0usize);

        let value_tensor = Tensor::from_cache(
            vec![self.shape[0], self.shape[1], num_topk],
            format!("{}.output_value.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::TopKSoftmax(TopKSoftmax::new(
            indices_ptr,
            self.data,
            sums_tensor.data,
            indice_ptr,
            value_tensor.data,
            output_sequences,
            self.shape[1],
            num_topk,
            // self.shape[2],
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
        sequence_chunk_size: usize,
        batch_size: usize,
        eps: T,
        scope_name: String,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> (Self, Self) {
        let output_hidden_tensor = Tensor::from_cache(
            vec![sequence_chunk_size, batch_size, word_embedding.shape[1]],
            format!("{}.output_hidden", scope_name),
            cache.clone(),
            operator_queue.clone(),
        );

        let output_normal_tensor = Tensor::from_cache(
            vec![sequence_chunk_size, batch_size, word_embedding.shape[1]],
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
}

unsafe impl<T: Copy + Default + Send + Sync> Send for Tensor<T> {}
unsafe impl<T: Copy + Default + Send + Sync> Sync for Tensor<T> {}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;
    use std::collections::HashMap;
    use std::f16;
    use std::thread;

    #[test]
    fn test_experts_softmax_norm_f32() {
        let cache: Rc<RefCell<Cache<f32>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f32>>>> = Rc::new(RefCell::new(Vec::new()));

        let sequence_chunk_size = 1;
        let batch_size = 2;
        let num_experts = 8;
        let num_experts_per_tok = 2;

        let shape = vec![sequence_chunk_size, batch_size, num_experts];
        let tensor = Tensor::<f32>::from_cache(
            shape.clone(),
            "model.layers.0.logits".to_string(),
            //"input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let data: Vec<f32> = vec![
            // token 0
            1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, // token 1
            5.0, 6.0, 7.0, 8.0, 4.5, 5.5, 6.5, 7.5,
        ];
        unsafe {
            tensor
                .data
                .copy_from_nonoverlapping(data.as_ptr(), data.len());
        }

        let (experts_indicator, indice_ptr, weight_ptr, topk_indices_ptr) = tensor
            .experts_softmax_norm(num_experts, num_experts_per_tok, "softmax_norm".to_string());

        for op in operator_queue.borrow_mut().iter() {
            op.run(0, sequence_chunk_size, batch_size, 1, 0);
        }

        let num_tokens = sequence_chunk_size * batch_size;
        let indices = unsafe { std::slice::from_raw_parts(indice_ptr, num_experts * num_tokens) };
        let weights = unsafe { std::slice::from_raw_parts(weight_ptr, num_experts * num_tokens) };
        let indicators = unsafe { std::slice::from_raw_parts(experts_indicator, num_experts) };

        // Token 0: input [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5], topk indices 3, 7
        let data0 = &data[0..num_experts];
        let exps0: Vec<f32> = data0.iter().map(|v| v.exp()).collect();
        let sum0: f32 = exps0.iter().sum();
        assert_ulps_eq!(weights[3 * num_tokens + 0], exps0[3] / sum0);
        assert_ulps_eq!(weights[7 * num_tokens + 0], exps0[7] / sum0);
        assert!(indices[3 * num_tokens + 0]);
        assert!(indices[7 * num_tokens + 0]);

        // Token 1: input [5.0, 6.0, 7.0, 8.0, 4.5, 5.5, 6.5, 7.5], topk indices 3, 7
        let data1 = &data[num_experts..2 * num_experts];
        let exps1: Vec<f32> = data1.iter().map(|v| v.exp()).collect();
        let sum1: f32 = exps1.iter().sum();
        assert_ulps_eq!(weights[3 * num_tokens + 1], exps1[3] / sum1);
        assert_ulps_eq!(weights[7 * num_tokens + 1], exps1[7] / sum1);
        assert!(indices[3 * num_tokens + 1]);
        assert!(indices[7 * num_tokens + 1]);

        // Check indicators
        assert!(!indicators[0]);
        assert!(!indicators[1]);
        assert!(!indicators[2]);
        assert!(indicators[3]);
        assert!(!indicators[4]);
        assert!(!indicators[5]);
        assert!(!indicators[6]);
        assert!(indicators[7]);

        // Check other indices and weights are false/zero
        for e in 0..num_experts {
            for t in 0..num_tokens {
                let offset = e * num_tokens + t;
                if (e == 3 && (t == 0 || t == 1)) || (e == 7 && (t == 0 || t == 1)) {
                    continue;
                }
                assert!(!indices[offset]);
                assert_ulps_eq!(weights[offset], 0.0);
            }
        }
    }

    #[test]
    fn test_experts_softmax_norm_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let sequence_chunk_size = 1;
        let batch_size = 2;
        let num_experts = 128;
        let num_experts_per_tok = 8;
        let thread_num = 8;

        let shape = vec![sequence_chunk_size, batch_size, num_experts];
        let tensor = Tensor::<f16>::from_cache(
            shape.clone(),
            "model.layers.0.logits".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let num_tokens = sequence_chunk_size * batch_size;
        let mut data_f32: Vec<f32> = Vec::with_capacity(num_tokens * num_experts);

        // Generate data
        for t in 0..num_tokens {
            for i in 0..num_experts {
                // Base value
                let v = ((i as f32 + t as f32 * 13.0) * 1.1) % 20.0 - 10.0;
                data_f32.push(v);
            }
            // Set top values explicitly to ensure stability
            let base = t * num_experts;
            for k in 0..num_experts_per_tok {
                // Set experts [0..k] to high values to ensure they are selected
                // Using distinct values to avoid sorting ambiguity
                data_f32[base + k] = 20.0 + (num_experts_per_tok - k) as f32;
            }
        }

        let data: Vec<f16> = data_f32.iter().map(|&x| x as f16).collect();
        unsafe {
            tensor
                .data
                .copy_from_nonoverlapping(data.as_ptr(), data.len());
        }

        let (experts_indicator, indice_ptr, weight_ptr, _topk_indices_ptr) = tensor
            .experts_softmax_norm(num_experts, num_experts_per_tok, "softmax_norm".to_string());

        for op in operator_queue.borrow_mut().iter() {
            for i in 0..thread_num {
                op.run(0, sequence_chunk_size, batch_size, thread_num, i);
            }
        }

        let indices = unsafe { std::slice::from_raw_parts(indice_ptr, num_experts * num_tokens) };
        let weights = unsafe { std::slice::from_raw_parts(weight_ptr, num_experts * num_tokens) };
        let indicators = unsafe { std::slice::from_raw_parts(experts_indicator, num_experts) };

        for t in 0..num_tokens {
            let start = t * num_experts;
            let end = start + num_experts;
            let token_input = &data_f32[start..end];

            let mut expected: Vec<(usize, f32)> = token_input.iter().copied().enumerate().collect();
            // Sort descending by value
            expected.sort_by(|a, b| b.1.total_cmp(&a.1));

            let topk = &expected[0..num_experts_per_tok];
            let max_val = topk.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = topk.iter().map(|v| (v.1 - max_val).exp()).sum();

            for i in 0..num_experts_per_tok {
                let (idx, val) = topk[i];
                let prob = (val - max_val).exp() / denom;

                assert!(
                    indicators[idx],
                    "Expert {} should be selected (token {})",
                    idx, t
                );

                let offset = idx * num_tokens + t;
                assert!(
                    indices[offset],
                    "Index ptr for expert {} token {} should be true",
                    idx, t
                );

                let weight = weights[offset];

                assert!(
                    ((weight as f32) - prob).abs() < 1e-3,
                    "Weight mismatch for expert {} token {}: expected {:?}, got {:?}",
                    idx,
                    t,
                    prob,
                    weight
                );
            }
        }
    }

    #[test]
    fn test_topk_softmax_f32() {
        let cache: Rc<RefCell<Cache<f32>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f32>>>> = Rc::new(RefCell::new(Vec::new()));

        let sequence_chunk_size = 1;
        let batch_size = 2;
        let num_topk = 8;
        let thread_num = 2;
        let num_candidates_per_thread = num_topk;
        let num_candidates = num_candidates_per_thread * thread_num;

        // Mock inputs for topk_softmax, which would come from matmul_local_topk
        // value_tensor shape: [sequence_chunk_size, batch_size, num_candidates_per_thread * thread_num]
        let value_shape = vec![sequence_chunk_size, batch_size, num_candidates];
        let value_tensor = Tensor::<f32>::from_cache(
            value_shape,
            "model.layers.0.values".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // sums_tensor shape: [sequence_chunk_size, batch_size, thread_num]
        let sums_shape = vec![sequence_chunk_size, batch_size, thread_num];
        let sums_tensor = Tensor::<f32>::from_cache(
            sums_shape,
            "model.layers.0.sums".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let mut output_sequences = vec![0usize; sequence_chunk_size * batch_size];

        // Data for token 0
        let values0: Vec<f32> = (0..num_candidates).map(|i| 5.0 - i as f32 * 0.1).collect();
        let indices0: Vec<usize> = (0..num_candidates).collect();
        // Data for token 1
        let values1: Vec<f32> = (0..num_candidates).map(|i| 8.0 - i as f32 * 0.2).collect();
        let indices1: Vec<usize> = (100..(100 + num_candidates)).collect();

        let mut all_values = Vec::new();
        all_values.extend_from_slice(&values0);
        all_values.extend_from_slice(&values1);

        let mut all_indices = Vec::new();
        all_indices.extend_from_slice(&indices0);
        all_indices.extend_from_slice(&indices1);

        let indices_ptr = all_indices.as_ptr();

        unsafe {
            value_tensor
                .data
                .copy_from_nonoverlapping(all_values.as_ptr(), all_values.len());
            // sums_tensor is not used in the current implementation of TopKSoftmax, so we can leave it as 0.
        }

        let (output_indices_ptr, output_value_tensor) = value_tensor.topk_softmax(
            indices_ptr,
            &sums_tensor,
            output_sequences.as_mut_ptr(),
            num_topk,
            "model.layers.0.topk_softmax".to_string(),
        );

        for i in 0..thread_num {
            println!("Running operator for thread {}", i);
            for op in operator_queue.borrow_mut().iter() {
                op.run(0, sequence_chunk_size, batch_size, thread_num, i);
                // op.run(0, sequence_chunk_size, batch_size, 2, 1);
            }
        }

        let num_tokens = sequence_chunk_size * batch_size;
        let output_indices =
            unsafe { std::slice::from_raw_parts(output_indices_ptr, num_tokens * num_topk) };
        let output_values =
            unsafe { std::slice::from_raw_parts(output_value_tensor.data, num_tokens * num_topk) };

        // Verification for token 0
        let mut candidates0: Vec<_> = indices0.iter().zip(values0.iter()).collect();
        candidates0.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values0: Vec<f32> = candidates0.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val0 = top_values0[0];
        let exps0: Vec<f32> = top_values0.iter().map(|v| (v - max_val0).exp()).collect();
        let sum_exps0: f32 = exps0.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[i], *candidates0[i].0);
            assert_ulps_eq!(output_values[i], exps0[i] / sum_exps0, max_ulps = 4);
        }

        // Verification for token 1
        let mut candidates1: Vec<_> = indices1.iter().zip(values1.iter()).collect();
        candidates1.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values1: Vec<f32> = candidates1.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val1 = top_values1[0];
        let exps1: Vec<f32> = top_values1.iter().map(|v| (v - max_val1).exp()).collect();
        let sum_exps1: f32 = exps1.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[num_topk + i], *candidates1[i].0);
            assert_ulps_eq!(
                output_values[num_topk + i],
                exps1[i] / sum_exps1,
                max_ulps = 4
            );
        }

        // Check output sequences (sampled tokens)
        assert_eq!(output_sequences[0], *candidates0[0].0);
        assert_eq!(output_sequences[1], *candidates1[0].0);
    }

    #[test]
    fn test_topk_softmax_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let sequence_chunk_size = 1;
        let batch_size = 2;
        let num_topk = 8;
        let thread_num = 2;
        let num_candidates_per_thread = num_topk;
        let num_candidates = num_candidates_per_thread * thread_num;

        // Mock inputs for topk_softmax
        let value_shape = vec![sequence_chunk_size, batch_size, num_candidates];
        let value_tensor = Tensor::<f16>::from_cache(
            value_shape,
            "model.layers.0.values".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let sums_shape = vec![sequence_chunk_size, batch_size, thread_num];
        let sums_tensor = Tensor::<f16>::from_cache(
            sums_shape,
            "model.layers.0.sums".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let mut output_sequences = vec![0usize; sequence_chunk_size * batch_size];

        // Data for token 0
        let values0: Vec<f32> = (0..num_candidates).map(|i| 5.0 - i as f32 * 0.1).collect();
        let indices0: Vec<usize> = (0..num_candidates).collect();
        // Data for token 1
        let values1: Vec<f32> = (0..num_candidates).map(|i| 8.0 - i as f32 * 0.2).collect();
        let indices1: Vec<usize> = (100..(100 + num_candidates)).collect();

        let mut all_values_f32 = Vec::new();
        all_values_f32.extend_from_slice(&values0);
        all_values_f32.extend_from_slice(&values1);
        let all_values: Vec<f16> = all_values_f32.iter().map(|&x| x as f16).collect();

        let mut all_indices = Vec::new();
        all_indices.extend_from_slice(&indices0);
        all_indices.extend_from_slice(&indices1);

        let indices_ptr = all_indices.as_ptr();

        unsafe {
            value_tensor
                .data
                .copy_from_nonoverlapping(all_values.as_ptr(), all_values.len());
        }

        let (output_indices_ptr, output_value_tensor) = value_tensor.topk_softmax(
            indices_ptr,
            &sums_tensor,
            output_sequences.as_mut_ptr(),
            num_topk,
            "model.layers.0.topk_softmax".to_string(),
        );

        for i in 0..thread_num {
            for op in operator_queue.borrow_mut().iter() {
                op.run(0, sequence_chunk_size, batch_size, thread_num, i);
            }
        }

        let num_tokens = sequence_chunk_size * batch_size;
        let output_indices =
            unsafe { std::slice::from_raw_parts(output_indices_ptr, num_tokens * num_topk) };
        let output_values =
            unsafe { std::slice::from_raw_parts(output_value_tensor.data, num_tokens * num_topk) };

        // Verification for token 0
        let mut candidates0: Vec<_> = indices0.iter().zip(values0.iter()).collect();
        candidates0.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values0: Vec<f32> = candidates0.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val0 = top_values0[0];
        let exps0: Vec<f32> = top_values0.iter().map(|v| (v - max_val0).exp()).collect();
        let sum_exps0: f32 = exps0.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[i], *candidates0[i].0);
            let val = (output_values[i] as f32);
            let expected = exps0[i] / sum_exps0;
            assert!(
                (val - expected).abs() < 1e-3,
                "Mismatch at token 0 index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        // Verification for token 1
        let mut candidates1: Vec<_> = indices1.iter().zip(values1.iter()).collect();
        candidates1.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values1: Vec<f32> = candidates1.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val1 = top_values1[0];
        let exps1: Vec<f32> = top_values1.iter().map(|v| (v - max_val1).exp()).collect();
        let sum_exps1: f32 = exps1.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[num_topk + i], *candidates1[i].0);
            let val = (output_values[num_topk + i] as f32);
            let expected = exps1[i] / sum_exps1;
            assert!(
                (val - expected).abs() < 1e-3,
                "Mismatch at token 1 index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        // Check output sequences (sampled tokens)
        assert_eq!(output_sequences[0], *candidates0[0].0);
        assert_eq!(output_sequences[1], *candidates1[0].0);
    }

    #[test]
    fn test_matmul3_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        // Dimensions matching test_kqv_f16_avx512_multi_tile in matmul3.rs
        // M = 12 (4 * MB=3)
        // K = 64
        // NQ = 96 (3 * NR=32)
        // NKV = 96
        // HEAD_DIM = 128
        let sequence_chunk_size = 1;
        let batch_size = 12;
        let hidden_size = 64;
        let q_dim = 96;
        let kv_dim = 96;
        let head_dim = 128;

        // Input: [sequence_chunk_size, batch_size, hidden_size]
        let input_shape = vec![sequence_chunk_size, batch_size, hidden_size];
        let input_tensor = Tensor::<f16>::from_cache(
            input_shape.clone(),
            "model.layers.0.input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // Weights: Tensor expects [N, K] shape for shape[0] usage, but MatMul3 expects KxN data layout.
        let q_weight_shape = vec![q_dim, hidden_size];
        let k_weight_shape = vec![kv_dim, hidden_size];
        let v_weight_shape = vec![kv_dim, hidden_size];

        let q_weight = Tensor::<f16>::from_cache(
            q_weight_shape.clone(),
            "q_weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );
        let k_weight = Tensor::<f16>::from_cache(
            k_weight_shape.clone(),
            "k_weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );
        let v_weight = Tensor::<f16>::from_cache(
            v_weight_shape.clone(),
            "v_weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // RoPE: [head_dim]
        let rope_shape = vec![head_dim];
        let position_embedding = Tensor::<f16>::from_cache(
            rope_shape.clone(),
            "rope.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // Initialize data matching test_kqv_f16_avx512_multi_tile
        let num_input = sequence_chunk_size * batch_size * hidden_size;
        let mut input_data = vec![0.0f16; num_input];
        for i in 0..batch_size {
            for k in 0..hidden_size {
                input_data[i * hidden_size + k] = (((i * 7 + k * 3) % 19) as f32 * 0.01) as f16;
            }
        }
        unsafe {
            input_tensor
                .data
                .copy_from_nonoverlapping(input_data.as_ptr(), num_input);
        }

        // Initialize Weights (KxN layout)
        let mut q_data = vec![0.0f16; q_dim * hidden_size];
        let mut k_data = vec![0.0f16; kv_dim * hidden_size];
        let mut v_data = vec![0.0f16; kv_dim * hidden_size];

        for k in 0..hidden_size {
            for n in 0..q_dim {
                q_data[k * q_dim + n] = (((k * 5 + n * 11) % 23) as f32 * 0.01) as f16;
            }
            for n in 0..kv_dim {
                k_data[k * kv_dim + n] = (((k * 3 + n * 7) % 29) as f32 * 0.01) as f16;
                v_data[k * kv_dim + n] = (((k * 9 + n * 4) % 31) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            q_weight
                .data
                .copy_from_nonoverlapping(q_data.as_ptr(), q_data.len());
            k_weight
                .data
                .copy_from_nonoverlapping(k_data.as_ptr(), k_data.len());
            v_weight
                .data
                .copy_from_nonoverlapping(v_data.as_ptr(), v_data.len());
        }

        // Initialize RoPE (All zeros in the reference test)
        let rope_data = vec![0.0f16; head_dim];
        unsafe {
            position_embedding
                .data
                .copy_from_nonoverlapping(rope_data.as_ptr(), head_dim);
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let (q_out, k_out, v_out) = input_tensor.matmul3(
            &q_weight,
            &k_weight,
            &v_weight,
            &position_embedding,
            head_dim,
            params,
            "model.layers.0.matmul3".to_string(),
        );

        // Run operators
        for op in operator_queue.borrow_mut().iter() {
            op.run(0, sequence_chunk_size, batch_size, 1, 0);
        }

        // Verification
        // Since NQ=96 < HEAD_DIM=128, finalize (RMS+RoPE) is NOT performed in MatMul3 logic for this test case.
        // So we just verify MatMul result.

        let verify_matmul =
            |output_tensor: &Tensor<f16>, weight_data_kxn: &[f16], n_dim: usize, name: &str| {
                let out_len = sequence_chunk_size * batch_size * n_dim;
                let out_data = unsafe { std::slice::from_raw_parts(output_tensor.data, out_len) };

                for i in 0..batch_size {
                    for j in 0..n_dim {
                        let mut sum = 0.0f32;
                        for k in 0..hidden_size {
                            let a_val = input_data[i * hidden_size + k] as f32;
                            let w_val = weight_data_kxn[k * n_dim + j] as f32;
                            sum += a_val * w_val;
                        }

                        let val = out_data[i * n_dim + j] as f32;
                        assert!(
                            (val - sum).abs() < 0.5, // Epsilon from test_kqv_f16_avx512_multi_tile is 5e-1
                            "{} mismatch at batch {}, col {}: got {}, expected {}",
                            name,
                            i,
                            j,
                            val,
                            sum
                        );
                    }
                }
            };

        verify_matmul(&q_out, &q_data, q_dim, "Q");
        verify_matmul(&k_out, &k_data, kv_dim, "K");
        verify_matmul(&v_out, &v_data, kv_dim, "V");
    }

    #[test]
    fn test_matmul_local_topk_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let sequence_chunk_size = 1;
        let batch_size = 12;
        let hidden_size = 64; // K
        let intermediate_size = 96; // N
        let topk = 10;

        // Use available threads but cap at 4 for test consistency
        // let max_threads = MatMulTopK::<f16>::detect_threads();
        let thread_num = 4;
        // .min(max_threads);

        let input_shape = vec![sequence_chunk_size, batch_size, hidden_size];
        let input_tensor = Tensor::<f16>::from_cache(
            input_shape,
            "model.layers.0.input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // Weight shape [N, K]
        let weight_shape = vec![intermediate_size, hidden_size];
        let weight_tensor = Tensor::<f16>::from_cache(
            weight_shape,
            "weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // Init data
        let m = sequence_chunk_size * batch_size;
        let k = hidden_size;
        let n = intermediate_size;

        let mut input_data = vec![0.0f16; m * k];
        for i in 0..m {
            for j in 0..k {
                input_data[i * k + j] = ((i + j) as f32 * 0.01) as f16;
            }
        }
        unsafe {
            input_tensor
                .data
                .copy_from_nonoverlapping(input_data.as_ptr(), m * k);
        }

        // Weight data K x N layout (MatMulTopK expects B to be KxN in memory)
        let mut weight_data = vec![0.0f16; k * n];
        for i in 0..k {
            for j in 0..n {
                weight_data[i * n + j] = ((i + j) as f32 * 0.001) as f16;
            }
        }
        unsafe {
            weight_tensor
                .data
                .copy_from_nonoverlapping(weight_data.as_ptr(), k * n);
        }

        let params = MatMulParams {
            a_row_step_macro: 6,   // MB
            b_row_step_macro: 64,  // NB
            column_step_macro: 64, // KC
            a_row_step_micro: 3,   // MR
            b_row_step_micro: 32,  // NR
        };

        let (indice_ptr, value_tensor) = input_tensor.matmul_local_topk(
            &weight_tensor,
            params,
            thread_num,
            topk,
            "model.layers.0.matmul_local_topk".to_string(),
        );

        // Run
        for op in operator_queue.borrow_mut().iter() {
            for i in 0..thread_num {
                op.run(0, sequence_chunk_size, batch_size, thread_num, i);
            }
        }

        // Verify
        // Output buffer size is based on max_threads, not thread_num
        let out_len = m * max_threads * topk;
        let indices = unsafe { std::slice::from_raw_parts(indice_ptr, out_len) };
        let values = unsafe { std::slice::from_raw_parts(value_tensor.data, out_len) };

        // Calculate expected
        for i in 0..m {
            // Full matmul for this row
            let mut row_c = vec![0.0f32; n];
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (input_data[i * k + kk] as f32) * (weight_data[kk * n + j] as f32);
                }
                row_c[j] = sum;
            }

            let mut indexed_row: Vec<(usize, f32)> = row_c.into_iter().enumerate().collect();
            indexed_row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let expected_topk = &indexed_row[0..topk];

            // Collect from all threads
            let mut merged: Vec<(usize, f32)> = Vec::new();
            for tid in 0..thread_num {
                let offset = i * (max_threads * topk) + tid * topk;
                for r in 0..topk {
                    merged.push((indices[offset + r], values[offset + r] as f32));
                }
            }
            merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let final_topk = &merged[0..topk];

            for r in 0..topk {
                let (exp_idx, exp_val) = expected_topk[r];
                let (got_idx, got_val) = final_topk[r];
                assert!(
                    (got_val - exp_val).abs() < 0.05,
                    "Value mismatch row {} rank {}: got {} exp {}",
                    i,
                    r,
                    got_val,
                    exp_val
                );
                if (got_val - exp_val).abs() < 1e-4 {
                    assert_eq!(got_idx, exp_idx, "Index mismatch row {} rank {}", i, r);
                }
            }
        }
    }

    #[test]
    fn test_matmul_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let sequence_chunk_size = 1;
        let batch_size = 12;
        let hidden_size = 64; // K
        let intermediate_size = 96; // N

        let input_shape = vec![sequence_chunk_size, batch_size, hidden_size];
        let input_tensor = Tensor::<f16>::from_cache(
            input_shape.clone(),
            "input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let weight_shape = vec![intermediate_size, hidden_size];
        let weight_tensor = Tensor::<f16>::from_cache(
            weight_shape.clone(),
            "weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let m = sequence_chunk_size * batch_size;
        let k = hidden_size;
        let n = intermediate_size;

        let mut input_data = vec![0.0f16; m * k];
        for i in 0..m {
            for j in 0..k {
                input_data[i * k + j] = (((i * 7 + j * 3) % 19) as f32 * 0.1) as f16;
            }
        }
        unsafe {
            input_tensor
                .data
                .copy_from_nonoverlapping(input_data.as_ptr(), m * k);
        }

        // Initialize Weight (K x N layout in memory)
        let mut weight_data = vec![0.0f16; k * n];
        for i in 0..k {
            for j in 0..n {
                weight_data[i * n + j] = (((i * 5 + j * 11) % 23) as f32 * 0.1) as f16;
            }
        }
        unsafe {
            weight_tensor
                .data
                .copy_from_nonoverlapping(weight_data.as_ptr(), k * n);
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let output_tensor = input_tensor.matmul(
            &weight_tensor,
            params,
            sequence_chunk_size,
            "matmul_test".to_string(),
        );

        for op in operator_queue.borrow_mut().iter() {
            op.run(0, sequence_chunk_size, batch_size, 1, 0);
        }

        let out_len = m * n;
        let output_data = unsafe { std::slice::from_raw_parts(output_tensor.data, out_len) };

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a = input_data[i * k + kk] as f32;
                    let b = weight_data[kk * n + j] as f32;
                    sum += a * b;
                }
                let val = output_data[i * n + j] as f32;
                assert!(
                    (val - sum).abs() < 0.5,
                    "Mismatch at batch {}, col {}: got {}, expected {}",
                    i,
                    j,
                    val,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_matmul_add_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let sequence_chunk_size = 1;
        let batch_size = 12;
        let hidden_size = 64; // K
        let intermediate_size = 96; // N

        let input_shape = vec![sequence_chunk_size, batch_size, hidden_size];
        let input_tensor = Tensor::<f16>::from_cache(
            input_shape.clone(),
            "input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let weight_shape = vec![intermediate_size, hidden_size];
        let weight_tensor = Tensor::<f16>::from_cache(
            weight_shape.clone(),
            "weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let bias_shape = vec![sequence_chunk_size, batch_size, intermediate_size];
        let bias_tensor = Tensor::<f16>::from_cache(
            bias_shape.clone(),
            "bias".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let m = sequence_chunk_size * batch_size;
        let k = hidden_size;
        let n = intermediate_size;

        // Init Input
        let mut input_data = vec![0.0f16; m * k];
        for i in 0..m {
            for j in 0..k {
                input_data[i * k + j] = (((i * 7 + j * 3) % 19) as f32 * 0.1) as f16;
            }
        }
        unsafe {
            input_tensor
                .data
                .copy_from_nonoverlapping(input_data.as_ptr(), m * k);
        }

        // Init Weight (K x N layout)
        let mut weight_data = vec![0.0f16; k * n];
        for i in 0..k {
            for j in 0..n {
                weight_data[i * n + j] = (((i * 5 + j * 11) % 23) as f32 * 0.1) as f16;
            }
        }
        unsafe {
            weight_tensor
                .data
                .copy_from_nonoverlapping(weight_data.as_ptr(), k * n);
        }

        // Init Bias
        let mut bias_data = vec![0.0f16; m * n];
        for i in 0..m {
            for j in 0..n {
                bias_data[i * n + j] = (((i * 2 + j * 5) % 17) as f32 * 0.1) as f16;
            }
        }
        unsafe {
            bias_tensor
                .data
                .copy_from_nonoverlapping(bias_data.as_ptr(), m * n);
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let output_tensor = input_tensor.matmul_add(
            &weight_tensor,
            &bias_tensor,
            params,
            "matmul_add_test".to_string(),
        );

        for op in operator_queue.borrow_mut().iter() {
            op.run(0, sequence_chunk_size, batch_size, 1, 0);
        }

        let out_len = m * n;
        let output_data = unsafe { std::slice::from_raw_parts(output_tensor.data, out_len) };

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a = input_data[i * k + kk] as f32;
                    let b = weight_data[kk * n + j] as f32;
                    sum += a * b;
                }
                sum += bias_data[i * n + j] as f32;

                let val = output_data[i * n + j] as f32;
                assert!(
                    (val - sum).abs() < 0.5,
                    "Mismatch at batch {}, col {}: got {}, expected {}",
                    i,
                    j,
                    val,
                    sum
                );
            }
        }
    }
}
