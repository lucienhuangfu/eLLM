use std::cell::RefCell;
// use std::iter::zip;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::init::tensor_utils::get_strides;
use super::super::memory::allocator::allocate_init;
use super::super::memory::cache::Cache;
use crate::init::matmul_params::MatmulParams;

use super::super::compiler::map::experts_softmax_norm::ExpertsSoftmaxNorm;
use super::super::compiler::map::lookup_rms_map::LookupRMSMap;
use super::super::compiler::map::rms_map::RMSMap;
use super::super::compiler::map::topk_softmax::TopKSoftmax;
// use super::super::compiler::mul::attention_mul_add::AttentionMul;
use super::super::compiler::mul::attention::Attention;
use super::super::compiler::mul::experts_matmul_mul::ExpertsMatmulMul;
use super::super::compiler::mul::experts_matmul_silu_mul_matmul::ExpertsMatmulSilu;
use super::super::compiler::mul::experts_merge_add::ExpertsMergeAdd;
use super::super::compiler::mul::matmul::Matmul;
use super::super::compiler::mul::matmul3::Matmul3;
use super::super::compiler::mul::matmul_add::MatmulAdd;
use super::super::compiler::mul::matmul_silu_mul_matmul::MatmulSilu;
use super::super::compiler::mul::matmul_topk::MatmulTopK;
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
        params: MatmulParams,
        scope_name: String,
    ) -> Self {
        // down_weights [num_experts, hidden_size, intermediate_size]
        // output [sequence_chunk_size, batch_size, num_experts_per_token, hidden_size]
        let output_shape = vec![self.shape[1], self.shape[2], num_experts_per_tok, down_weights.shape[1]];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::ExpertsMatmulMul(ExpertsMatmulMul::new(
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
        params: MatmulParams,
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

        let operator = Operator::ExpertsMatmulSiluMulMatmul(ExpertsMatmulSilu::new(
            self.data,
            gate_weights.data,
            up_weights.data,
            experts_indicator,
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
        let mut topk_indices_ptr = unsafe { allocate_init(num_experts_per_tok * self.shape[0] * self.shape[1]  , 0usize) };
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
        params: MatmulParams,
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
        println!( "Before matmul operator creation in Tensor matmul: {}", scope_name);

        let operator = unsafe {
            Operator::Matmul(Matmul::new(
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
        println!( "After matmul in Tensor matmul: {}", scope_name);

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn matmul_add(
        &self,
        tensor2: &Tensor<T>,
        tensor3: &Tensor<T>,
        params: MatmulParams,
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

        let operator = Operator::MatmulAdd(MatmulAdd::new(
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

    /* 
    pub fn matmul3(
        &self,
        q_weight: &Tensor<T>,
        k_weight: &Tensor<T>,
        v_weight: &Tensor<T>,
        position_embedding: &Tensor<T>,
        head_dim: usize,
        params: MatmulParams,
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

        let operator = Operator::Matmul3(Matmul3::new(
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
    }*/

    pub fn matmul_silu_mul_matmul(
        &self,
        tensor2: &Tensor<T>,
        tensor3: &Tensor<T>,
        params: MatmulParams,
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

        let operator = Operator::MatmulSiluMulMatmul(MatmulSilu::new(
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
        params: MatmulParams,
        thread_num: usize,
        scope_name: String,
    ) -> (*const usize, Self, Self) {
        let a_row = self.shape[1];
        let b_row = tensor2.shape[1];
        let column = self.shape[2];

        let output_shape = vec![self.shape[0], self.shape[1], tensor2.shape[0]];

        let indice_ptr = allocate_init(output_shape.iter().product(), 0usize);

        let value_tensor = Tensor::<T>::from_cache(
            output_shape.clone(),
            format!("{}.values.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let sum_tensor = Tensor::from_cache(
            vec![self.shape[0], self.shape[1], thread_num],
            format!("{}.sums.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::MatmulTopK(MatmulTopK::new(
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
    use std::collections::HashMap;
    use std::thread;

    // use num_cpus;
    // use std::sync::Arc;
    // use std::sync::Barrier;
    use super::*;

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

        let (experts_indicator, indice_ptr, weight_ptr, topk_indices_ptr) = tensor.experts_softmax_norm(
            num_experts,
            num_experts_per_tok,
            "softmax_norm".to_string(),
        );

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

        for op in operator_queue.borrow_mut().iter() {
            op.run(0, sequence_chunk_size, batch_size, 2, 0);
            op.run(0, sequence_chunk_size, batch_size, 2, 1);
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
}
