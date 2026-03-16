use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::common::num_traits::Sigmoid;
use crate::common::num_traits::Sqrt;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};

use crate::common::matmul_params::MatMulParams;
use crate::common::tensor_utils::get_strides;
use crate::mem_mgr::allocator::allocate_init;
use crate::mem_mgr::cache::Cache;

use crate::operators::movement::LiftVector;
use crate::operators::routing::ExpertsSoftmaxNorm;
use crate::operators::transform::LookupRMSMap;

use crate::common::sequence_slice::SequenceSlice;
use crate::operators::expert::{ExpertsMatMulDown, ExpertsMatMulSilu, ExpertsMergeAdd};
use crate::operators::linear::{Attention, MatMul, MatMul3, MatMulAdd};
use crate::operators::routing::TopKSoftmax;
use crate::runtime::inference::{Phase, SequenceState};
// use super::super::operators::mul::matmul_silu_mul_matmul::MatMulSilu;
use crate::operators::routing::MatMulTopK;
use crate::operators::transform::AddRMSZipMap;
use crate::operators::transform::AddZipMap;
use crate::runtime::operator::Operator;
// use super::super::operators::zip_map::complex_zip::ComplexZipMap;
// use super::super::operators::zip_map::silu_mul_zip::SiluMulZipMap;
use crate::operators::transform::RMSMap;

#[derive(Clone)]
pub struct Tensor<T>
where
    T: Copy + PartialOrd,
{
    pub data: *mut T,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub tensor_name: String,
    pub cache: Rc<RefCell<Cache<T>>>,
    pub operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    // pub size: usize,
    // pub is_contiguous: bool,
}

#[derive(Clone)]
pub struct TensorCtx<T>
where
    T: Copy + PartialOrd,
{
    pub cache: Rc<RefCell<Cache<T>>>,
    pub operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> TensorCtx<T>
where
    T: Copy + PartialOrd,
{
    pub fn new(
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        Self {
            cache,
            operator_queue,
        }
    }

    pub fn take_operator_queue(&self) -> Vec<Operator<T>> {
        std::mem::take(&mut *self.operator_queue.borrow_mut())
    }
}

impl<T> TensorCtx<T>
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
    pub fn tensor(&self, shape: Vec<usize>, tensor_name: String) -> Tensor<T> {
        Tensor::from_cache(
            shape,
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        )
    }

    pub fn tensor_from_vec(
        &self,
        shape: Vec<usize>,
        data: Vec<T>,
        tensor_name: String,
    ) -> Tensor<T> {
        Tensor::from_vec(
            shape,
            data,
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        )
    }

    pub fn zeros(&self, shape: Vec<usize>, tensor_name: String) -> Tensor<T> {
        Tensor::zeros(
            shape,
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        )
    }

    pub fn lookup_rms(
        &self,
        sequences_ptr: *const usize,
        word_embedding: &Tensor<T>,
        batch_size: usize,
        eps: T,
        scope_name: String,
    ) -> (Tensor<T>, Tensor<T>) {
        Tensor::lookup_rms(
            sequences_ptr,
            word_embedding,
            batch_size,
            eps,
            scope_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        )
    }
}

impl<T> Tensor<T>
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
            // self.shape[0],
            self.shape[1],
            self.shape[2],
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
            // self.shape[0],
            self.shape[1],
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
        inverse_sqrt_head: T,
        decode_only_flag: bool,
        thread_num: usize,
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
            output_tensor.data,
            self.shape[0],
            self.shape[1],
            k_tensor.shape[1],
            k_tensor.shape[2],
            1,
            8,
            self.shape[2],
            inverse_sqrt_head,
            decode_only_flag,
            thread_num,
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
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        // output [batch_size, hidden_size]
        let output_shape = vec![self.shape[0], self.shape[2]];

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
            1,
            num_experts,
            self.shape[0],
            self.shape[1],
            self.shape[2],
            false,
            decode_only_flag,
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
        num_experts_per_tok: usize,
        params: MatMulParams,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        // down_weights [num_experts, hidden_size, intermediate_size]
        // output [batch_size, num_experts_per_token, hidden_size]
        let output_shape = vec![self.shape[1], num_experts_per_tok, down_weights.shape[1]];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::ExpertsMatMulDown(unsafe {
            ExpertsMatMulDown::new(
                self.data,
                down_weights.data,
                experts_indicator,
                indice_ptr,
                weight_ptr,
                topk_indices_ptr,
                output_tensor.data,
                down_weights.shape[0],
                self.shape[1],
                down_weights.shape[2],
                down_weights.shape[1],
                num_experts_per_tok,
                params,
                decode_only_flag,
            )
        });

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
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        // gate_weights [num_experts, intermediate_size, hidden_size]
        // output [num_experts, batch_size, intermediate_size]
        let output_shape = vec![gate_weights.shape[0], self.shape[0], gate_weights.shape[1]];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::ExpertsMatMulSilu(unsafe {
            ExpertsMatMulSilu::new(
                self.data,
                gate_weights.data,
                up_weights.data,
                experts_indicator,
                indice_ptr,
                output_tensor.data,
                self.shape[0],
                gate_weights.shape[1],
                self.shape[1],
                gate_weights.shape[0],
                params.a_row_step_macro,
                params.b_row_step_macro,
                params.column_step_macro,
                params.a_row_step_micro,
                params.b_row_step_micro,
                decode_only_flag,
            )
        });

        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }

    pub fn experts_softmax_norm(
        &self,
        num_experts: usize,
        num_experts_per_tok: usize,
        decode_only_flag: bool,
        scope_name: String,
    ) -> (*mut bool, *mut bool, *mut T, *mut usize) {
        // [(experts_id, [(token_id, weight)])]
        // sorted_ids: Vec<(usize, Vec<(usize, T)>)>,

        // [expert_num] bool
        let experts_indicator = unsafe { allocate_init(num_experts, false) };
        // [expert_num, batch_size] indice bool vec<bool>
        // [expert_num, batch_size] weight f16
        let length = num_experts * self.shape[0];
        let indice_ptr = unsafe { allocate_init(length, false) };
        let weight_ptr = unsafe { allocate_init(length, T::default()) };
        let mut topk_indices_ptr =
            unsafe { allocate_init(num_experts_per_tok * self.shape[0], 0usize) };
        // vec![0usize; num_experts * self.shape[0]];

        let operator = Operator::ExpertsSoftmaxNorm(ExpertsSoftmaxNorm::new(
            self.data,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            topk_indices_ptr,
            self.shape[0],
            num_experts,
            num_experts_per_tok,
            decode_only_flag,
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

    pub fn from_vec(
        shape: Vec<usize>,
        data: Vec<T>,
        tensor_name: String,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let length: usize = shape.iter().product();
        assert!(
            data.len() == length,
            "Tensor::from_vec length mismatch: shape product {} != data len {}",
            length,
            data.len()
        );
        let v = Self::from_cache(shape, tensor_name, cache, operator_queue);
        unsafe {
            v.data.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        v
    }

    pub fn matmul(
        &self,
        tensor2: &Tensor<T>,
        params: MatMulParams,
        sequence_length: usize,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        let output_shape = vec![self.shape[0], tensor2.shape[0]];

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
                self.shape[0],
                tensor2.shape[0],
                self.shape[1],
                decode_only_flag,
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
        let output_shape = vec![self.shape[0], tensor2.shape[0]];

        let a_row = self.shape[0];
        let b_row = tensor2.shape[0];
        let column = self.shape[1];

        let output_tensor = Tensor::from_cache(
            output_shape.clone(),
            tensor_name,
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::MatMulAdd(unsafe {
            MatMulAdd::new(
                self.data,
                tensor2.data,
                tensor3.data,
                output_tensor.data,
                params,
                a_row,
                b_row,
                column,
            )
        });

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
        let a_h_row = self.shape[0];
        let col = self.shape[1];
        // let b_row = q_weight.shape[1];

        let q_state = Tensor::from_cache(
            vec![self.shape[0], q_weight.shape[0]],
            format!("{}.q_state", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let k_state = Tensor::from_cache(
            vec![self.shape[0], k_weight.shape[0]],
            format!("{}.k_state", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let v_state = Tensor::from_cache(
            vec![self.shape[0], v_weight.shape[0]],
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

    /*
    pub fn matmul_silu_mul_matmul(
        &self,
        tensor2: &Tensor<T>,
        tensor3: &Tensor<T>,
        params: MatMulParams,
        tensor_name: String,
    ) -> Self {
        // hidden_tensor [batch_size, hidden_size]
        // tensor2 [intermediate_size, hidden_size]
        // output [batch_size, intermediate_size]

        let a_row = self.shape[0];
        let b_row = tensor2.shape[0];
        let column = self.shape[1];

        let output_shape = vec![a_row, b_row];

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
    } */

    pub fn matmul_local_topk(
        &self,
        tensor2: &Tensor<T>,
        params: MatMulParams,
        // thread_num: usize,
        topk: usize,
        scope_name: String,
    ) -> (*const usize, Self) {
        let n = tensor2.shape[0];
        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let m = self.shape[0];
        let k = self.shape[1];
        let output_shape = vec![self.shape[0], thread_num * topk];

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

    pub fn topk_softmax(
        &self,
        indices_ptr: *const usize,
        // sums_tensor: &Tensor<T>,
        output_sequences: *mut usize,
        num_topk: usize,
        eos_id: usize,
        scope_name: String,
    ) -> (*const usize, Self) {
        let output_shape = vec![self.shape[0], num_topk];
        let indice_ptr = allocate_init(output_shape.iter().product(), 0usize);

        let value_tensor = Tensor::from_cache(
            vec![self.shape[0], num_topk],
            format!("{}.output_value.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::TopKSoftmax(TopKSoftmax::new(
            indices_ptr,
            self.data,
            indice_ptr,
            value_tensor.data,
            output_sequences,
            self.shape[0],
            num_topk,
            eos_id,
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
        sequences_ptr: *const usize,
        word_embedding: &Tensor<T>,
        batch_size: usize,
        eps: T,
        scope_name: String,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> (Self, Self) {
        let output_hidden_tensor = Tensor::from_cache(
            vec![batch_size, word_embedding.shape[1]],
            format!("{}.output_hidden", scope_name),
            cache.clone(),
            operator_queue.clone(),
        );

        let output_normal_tensor = Tensor::from_cache(
            vec![batch_size, word_embedding.shape[1]],
            format!("{}.output_normal", scope_name),
            cache.clone(),
            operator_queue.clone(),
        );

        let operator = Operator::LookupRMSMap(LookupRMSMap::new(
            sequences_ptr,
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

    pub fn lift_vector(&self) {
        let operator = Operator::LiftVector(LiftVector::new(self.data, self.shape[1]));
        self.operator_queue.borrow_mut().push(operator);
    }

    pub fn rms(&self, eps: T, decode_only_flag: bool, scope_name: String) -> Self {
        let output_tensor = Tensor::<T>::from_cache(
            self.shape.clone(),
            format!("{}.output", scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        let operator = Operator::RMSMap(RMSMap::new(
            self.data,
            output_tensor.data,
            // self.shape[0],
            self.shape[1],
            eps,
            decode_only_flag,
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }
}

unsafe impl<T: Copy + Default + Send + Sync + PartialOrd> Send for Tensor<T> {}
unsafe impl<T: Copy + Default + Send + Sync + PartialOrd> Sync for Tensor<T> {}

#[cfg(test)]
mod test {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_ulps_eq};
    use std::collections::HashMap;
    use std::f16;
    use std::mem;

    // ============================================================
    // helpers
    // ============================================================

    fn avail_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    #[inline]
    fn f32_from_f16(x: f16) -> f32 {
        // bitcast based f16->f32 (与你原实现一致)
        let bits: u16 = unsafe { mem::transmute(x) };
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits & 0x7C00) >> 10;
        let mant = bits & 0x03FF;

        let f_bits: u32 = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                let mut m = mant as u32;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let exp_f = (e + 127) as u32;
                sign | (exp_f << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            let exp_f = 0xFFu32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        } else {
            let exp_f = (exp as i32 - 15 + 127) as u32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        };

        f32::from_bits(f_bits)
    }

    fn run_operator_all_threads(
        op: &Operator<f16>,
        prefill_size: usize,
        decode_size: usize,
        cpu_num: usize,
    ) {
        for tid in 0..cpu_num {
            op.run(
                prefill_size,
                decode_size,
                cpu_num,
                tid,
                &[],
                &[],
                &[],
                &mut Vec::new(),
            );
        }
    }

    #[inline]
    fn silu_f32(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // NT layout helpers:
    // B_nt is N×K row-major => b_nt[j*K + kk]
    #[inline]
    fn idx_b_nt(j: usize, kk: usize, k: usize) -> usize {
        j * k + kk
    }

    // ============================================================
    // TopKSoftmax tests (unchanged - not matmul RHS related)
    // ============================================================

    #[test]
    fn test_topk_softmax_f32() {
        let cache: Rc<RefCell<Cache<f32>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f32>>>> = Rc::new(RefCell::new(Vec::new()));

        let batch_size = 2;
        let num_topk = 8;
        let thread_num = 2;
        let num_candidates_per_thread = num_topk;
        let num_candidates = num_candidates_per_thread * thread_num;
        let eos_id = 100;

        let value_shape = vec![batch_size, num_candidates];
        let value_tensor = Tensor::<f32>::from_cache(
            value_shape,
            "model.layers.0.values".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let sums_shape = vec![batch_size, thread_num];
        let sums_tensor = Tensor::<f32>::from_cache(
            sums_shape,
            "model.layers.0.sums".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let mut output_sequences = vec![0usize; batch_size * 2];

        let values0: Vec<f32> = (0..num_candidates).map(|i| 5.0 - i as f32 * 0.1).collect();
        let indices0: Vec<usize> = (0..num_candidates).collect();

        let values1: Vec<f32> = (0..num_candidates).map(|i| 8.0 - i as f32 * 0.2).collect();
        let indices1: Vec<usize> = (100..(100 + num_candidates)).collect();

        let mut all_values = Vec::new();
        all_values.extend_from_slice(&values0);
        all_values.extend_from_slice(&values1);

        let mut all_indices = Vec::new();
        all_indices.extend_from_slice(&indices0);
        all_indices.extend_from_slice(&indices1);

        let indices_ptr = all_indices.as_ptr();

        let mut batch_list = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            batch_list.push(SequenceState {
                filling_length: 0,
                sequence_index: 0,
                kv_index: 0,
                phase: Phase::Decode,
                // prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            });
        }
        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut decode_lists = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for batch_index in start..end {
                slices.push(SequenceSlice {
                    batch_index,
                    sequence_index: 0,
                    token_start_index: batch_index,
                    lift_index: 0,
                    length: 1,
                });
            }
            decode_lists.push(slices);
        }

        unsafe {
            value_tensor
                .data
                .copy_from_nonoverlapping(all_values.as_ptr(), all_values.len());
            let _ = sums_tensor; // sums currently unused
        }

        let (output_indices_ptr, output_value_tensor) = value_tensor.topk_softmax(
            indices_ptr,
            output_sequences.as_mut_ptr(),
            num_topk,
            eos_id,
            "model.layers.0.topk_softmax".to_string(),
        );

        for i in 0..thread_num {
            for op in operator_queue.borrow_mut().iter() {
                if let Operator::TopKSoftmax(operator) = op {
                    operator.run(
                        batch_size,
                        1,
                        thread_num,
                        i,
                        &decode_lists[i],
                        &mut batch_list,
                    );
                } else {
                    op.run(batch_size, 1, thread_num, i, &[], &[], &[], &mut Vec::new());
                }
            }
        }

        let num_tokens = batch_size;
        let output_indices =
            unsafe { std::slice::from_raw_parts(output_indices_ptr, num_tokens * num_topk) };
        let output_values =
            unsafe { std::slice::from_raw_parts(output_value_tensor.data, num_tokens * num_topk) };

        // token 0
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

        // token 1
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

        let batch_size = 2;
        let num_topk = 8;
        let thread_num = 2;
        let num_candidates_per_thread = num_topk;
        let num_candidates = num_candidates_per_thread * thread_num;
        let eos_id = 100;

        let value_shape = vec![batch_size, num_candidates];
        let value_tensor = Tensor::<f16>::from_cache(
            value_shape,
            "model.layers.0.values".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let sums_shape = vec![batch_size, thread_num];
        let sums_tensor = Tensor::<f16>::from_cache(
            sums_shape,
            "model.layers.0.sums".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let mut output_sequences = vec![0usize; batch_size];

        let values0: Vec<f32> = (0..num_candidates).map(|i| 5.0 - i as f32 * 0.1).collect();
        let indices0: Vec<usize> = (0..num_candidates).collect();

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

        let mut batch_list = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            batch_list.push(SequenceState {
                filling_length: 0,
                sequence_index: 0,
                kv_index: 0,
                phase: Phase::Decode,
                // prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            });
        }
        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut decode_lists = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for batch_index in start..end {
                slices.push(SequenceSlice {
                    batch_index,
                    sequence_index: 0,
                    token_start_index: batch_index,
                    lift_index: 0,
                    length: 1,
                });
            }
            decode_lists.push(slices);
        }

        unsafe {
            value_tensor
                .data
                .copy_from_nonoverlapping(all_values.as_ptr(), all_values.len());
            let _ = sums_tensor; // unused
        }

        let (output_indices_ptr, output_value_tensor) = value_tensor.topk_softmax(
            indices_ptr,
            output_sequences.as_mut_ptr(),
            num_topk,
            eos_id,
            "model.layers.0.topk_softmax".to_string(),
        );

        for i in 0..thread_num {
            for op in operator_queue.borrow_mut().iter() {
                if let Operator::TopKSoftmax(operator) = op {
                    operator.run(
                        batch_size,
                        1,
                        thread_num,
                        i,
                        &decode_lists[i],
                        &mut batch_list,
                    );
                } else {
                    op.run(batch_size, 1, thread_num, i, &[], &[], &[], &mut Vec::new());
                }
            }
        }

        let num_tokens = batch_size;
        let output_indices =
            unsafe { std::slice::from_raw_parts(output_indices_ptr, num_tokens * num_topk) };
        let output_values =
            unsafe { std::slice::from_raw_parts(output_value_tensor.data, num_tokens * num_topk) };

        // token 0
        let mut candidates0: Vec<_> = indices0.iter().zip(values0.iter()).collect();
        candidates0.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values0: Vec<f32> = candidates0.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val0 = top_values0[0];
        let exps0: Vec<f32> = top_values0.iter().map(|v| (v - max_val0).exp()).collect();
        let sum_exps0: f32 = exps0.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[i], *candidates0[i].0);
            let val = output_values[i] as f32;
            let expected = exps0[i] / sum_exps0;
            assert!(
                (val - expected).abs() < 1e-3,
                "Mismatch at token 0 index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        // token 1
        let mut candidates1: Vec<_> = indices1.iter().zip(values1.iter()).collect();
        candidates1.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values1: Vec<f32> = candidates1.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val1 = top_values1[0];
        let exps1: Vec<f32> = top_values1.iter().map(|v| (v - max_val1).exp()).collect();
        let sum_exps1: f32 = exps1.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[num_topk + i], *candidates1[i].0);
            let val = output_values[num_topk + i] as f32;
            let expected = exps1[i] / sum_exps1;
            assert!(
                (val - expected).abs() < 1e-3,
                "Mismatch at token 1 index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        assert_eq!(output_sequences[0], *candidates0[0].0);
        assert_eq!(output_sequences[1], *candidates1[0].0);
    }

    // ============================================================
    // MatMul3 tests — weights now NT (N×K row-major)
    // ============================================================

    #[test]
    fn test_matmul3_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let batch_size = 3;
        let hidden_size = 64;
        let q_dim = 96;
        let kv_dim = 96;
        let head_dim = 128;

        let input_shape = vec![batch_size, hidden_size];
        let input_tensor = Tensor::<f16>::from_cache(
            input_shape.clone(),
            "model.layers.0.input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // Tensor shape is [N, K] but raw mem_mgr should be NT: N×K
        let q_weight = Tensor::<f16>::from_cache(
            vec![q_dim, hidden_size],
            "q.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );
        let k_weight = Tensor::<f16>::from_cache(
            vec![kv_dim, hidden_size],
            "k.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );
        let v_weight = Tensor::<f16>::from_cache(
            vec![kv_dim, hidden_size],
            "v.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let position_embedding = Tensor::<f16>::from_cache(
            vec![head_dim],
            "rope.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // input
        let num_input = batch_size * hidden_size;
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

        // weights: NT (N×K)
        let mut q_data_nt = vec![0.0f16; q_dim * hidden_size];
        let mut k_data_nt = vec![0.0f16; kv_dim * hidden_size];
        let mut v_data_nt = vec![0.0f16; kv_dim * hidden_size];

        for n in 0..q_dim {
            for k in 0..hidden_size {
                // 原来是 [k*q_dim + n]，现在改为 [n*hidden + k]
                q_data_nt[n * hidden_size + k] = (((k * 5 + n * 11) % 23) as f32 * 0.01) as f16;
            }
        }
        for n in 0..kv_dim {
            for k in 0..hidden_size {
                k_data_nt[n * hidden_size + k] = (((k * 3 + n * 7) % 29) as f32 * 0.01) as f16;
                v_data_nt[n * hidden_size + k] = (((k * 9 + n * 4) % 31) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            q_weight
                .data
                .copy_from_nonoverlapping(q_data_nt.as_ptr(), q_data_nt.len());
            k_weight
                .data
                .copy_from_nonoverlapping(k_data_nt.as_ptr(), k_data_nt.len());
            v_weight
                .data
                .copy_from_nonoverlapping(v_data_nt.as_ptr(), v_data_nt.len());
        }

        let rope_data = vec![0.0f16; head_dim];
        unsafe {
            position_embedding
                .data
                .copy_from_nonoverlapping(rope_data.as_ptr(), head_dim);
        }

        let params = MatMulParams {
            a_row_step_macro: 3,
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

        for op in operator_queue.borrow_mut().iter() {
            op.run(batch_size, 1, 1, 0, &[], &[], &[], &mut Vec::new());
        }

        let verify_matmul_nt =
            |output_tensor: &Tensor<f16>, w_nt: &[f16], n_dim: usize, name: &str| {
                let out_len = batch_size * n_dim;
                let out_data = unsafe { std::slice::from_raw_parts(output_tensor.data, out_len) };

                for i in 0..batch_size {
                    for j in 0..n_dim {
                        let mut sum = 0.0f32;
                        for k in 0..hidden_size {
                            let a_val = input_data[i * hidden_size + k] as f32;
                            let w_val = w_nt[j * hidden_size + k] as f32; // NT
                            sum += a_val * w_val;
                        }

                        let val = out_data[i * n_dim + j] as f32;
                        assert!(
                            (val - sum).abs() < 0.5,
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

        verify_matmul_nt(&q_out, &q_data_nt, q_dim, "Q");
        verify_matmul_nt(&k_out, &k_data_nt, kv_dim, "K");
        verify_matmul_nt(&v_out, &v_data_nt, kv_dim, "V");
    }

    #[test]
    fn test_tensor_matmul3_f16_seq1_batch24() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let batch_size = 24;
        let hidden_size = 64;

        let q_dim = 96;
        let kv_dim = 96;
        let head_dim = 128;

        let input_tensor = Tensor::<f16>::from_cache(
            vec![batch_size, hidden_size],
            "model.layers.0.input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let q_weight = Tensor::<f16>::from_cache(
            vec![q_dim, hidden_size],
            "q.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );
        let k_weight = Tensor::<f16>::from_cache(
            vec![kv_dim, hidden_size],
            "k.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );
        let v_weight = Tensor::<f16>::from_cache(
            vec![kv_dim, hidden_size],
            "v.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let position_embedding = Tensor::<f16>::from_cache(
            vec![head_dim],
            "rope.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // input init
        let m = batch_size;
        let mut input_data = vec![0.0f16; m * hidden_size];
        for b in 0..batch_size {
            for kk in 0..hidden_size {
                input_data[b * hidden_size + kk] = (((b * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
            }
        }
        unsafe {
            input_tensor
                .data
                .copy_from_nonoverlapping(input_data.as_ptr(), input_data.len());
        }

        // weights init: NT (N×K)
        let mut q_data_nt = vec![0.0f16; q_dim * hidden_size];
        let mut k_data_nt = vec![0.0f16; kv_dim * hidden_size];
        let mut v_data_nt = vec![0.0f16; kv_dim * hidden_size];

        for n in 0..q_dim {
            for kk in 0..hidden_size {
                q_data_nt[n * hidden_size + kk] = (((kk * 5 + n * 11) % 23) as f32 * 0.01) as f16;
            }
        }
        for n in 0..kv_dim {
            for kk in 0..hidden_size {
                k_data_nt[n * hidden_size + kk] = (((kk * 3 + n * 7) % 29) as f32 * 0.01) as f16;
                v_data_nt[n * hidden_size + kk] = (((kk * 9 + n * 4) % 31) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            q_weight
                .data
                .copy_from_nonoverlapping(q_data_nt.as_ptr(), q_data_nt.len());
            k_weight
                .data
                .copy_from_nonoverlapping(k_data_nt.as_ptr(), k_data_nt.len());
            v_weight
                .data
                .copy_from_nonoverlapping(v_data_nt.as_ptr(), v_data_nt.len());
        }

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

        assert_eq!(q_out.shape, vec![batch_size, q_dim]);
        assert_eq!(k_out.shape, vec![batch_size, kv_dim]);
        assert_eq!(v_out.shape, vec![batch_size, kv_dim]);
        assert_eq!(operator_queue.borrow().len(), 1);
        assert!(matches!(&operator_queue.borrow()[0], Operator::MatMul3(_)));

        for op in operator_queue.borrow_mut().iter() {
            op.run(batch_size, 1, 1, 0, &[], &[], &[], &mut Vec::new());
        }

        let q_len = m * q_dim;
        let kv_len = m * kv_dim;
        let q_got = unsafe { std::slice::from_raw_parts(q_out.data, q_len) };
        let k_got = unsafe { std::slice::from_raw_parts(k_out.data, kv_len) };
        let v_got = unsafe { std::slice::from_raw_parts(v_out.data, kv_len) };

        let check_nt = |got: &[f16], w_nt: &[f16], n_dim: usize, name: &str| {
            for i in 0..m {
                for j in 0..n_dim {
                    let mut sum = 0.0f32;
                    for kk in 0..hidden_size {
                        sum += (input_data[i * hidden_size + kk] as f32)
                            * (w_nt[j * hidden_size + kk] as f32);
                    }
                    let val = got[i * n_dim + j] as f32;
                    assert!(
                        (val - sum).abs() < 0.5,
                        "{} mismatch at row {}, col {}: got {}, expected {}",
                        name,
                        i,
                        j,
                        val,
                        sum
                    );
                }
            }
        };

        check_nt(q_got, &q_data_nt, q_dim, "Q");
        check_nt(k_got, &k_data_nt, kv_dim, "K");
        check_nt(v_got, &v_data_nt, kv_dim, "V");
    }

    // ============================================================
    // MatMulTopK / matmul_local_topk — B is NT now
    // ============================================================

    #[test]
    fn test_matmul_local_topk_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let batch_size = 12;
        let hidden_size = 64; // K
        let intermediate_size = 96; // N
        let topk = 10;

        let thread_num = avail_threads();

        let input_tensor = Tensor::<f16>::from_cache(
            vec![batch_size, hidden_size],
            "model.layers.0.input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let weight_tensor = Tensor::<f16>::from_cache(
            vec![intermediate_size, hidden_size],
            "weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let m = batch_size;
        let k = hidden_size;
        let n = intermediate_size;

        // A
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

        // B_nt: N×K
        let mut weight_data_nt = vec![0.0f16; n * k];
        for j in 0..n {
            for kk in 0..k {
                // 原来 (kk + j)*0.001 写在 KxN，现在写在 NT
                weight_data_nt[j * k + kk] = ((kk + j) as f32 * 0.001) as f16;
            }
        }
        unsafe {
            weight_tensor
                .data
                .copy_from_nonoverlapping(weight_data_nt.as_ptr(), n * k);
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let (indice_ptr, value_tensor) = input_tensor.matmul_local_topk(
            &weight_tensor,
            params,
            topk,
            "model.layers.0.matmul_local_topk".to_string(),
        );

        for op in operator_queue.borrow_mut().iter() {
            for i in 0..thread_num {
                op.run(m, 1, thread_num, i, &[], &[], &[], &mut Vec::new());
            }
        }

        let out_len = m * thread_num * topk;
        let indices = unsafe { std::slice::from_raw_parts(indice_ptr, out_len) };
        let values = unsafe { std::slice::from_raw_parts(value_tensor.data, out_len) };

        for i in 0..m {
            // full reference C row (f32)
            let mut row_c = vec![0.0f32; n];
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (input_data[i * k + kk] as f32) * (weight_data_nt[j * k + kk] as f32);
                }
                row_c[j] = sum;
            }

            let mut indexed_row: Vec<(usize, f32)> = row_c.into_iter().enumerate().collect();
            indexed_row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let expected_topk = &indexed_row[0..topk];

            let mut merged: Vec<(usize, f32)> = Vec::new();
            for tid in 0..thread_num {
                let offset = i * (thread_num * topk) + tid * topk;
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
    fn test_matmul_local_topk_f16_no_ties_stable() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let batch_size = 12;
        let k = 64;
        let n = 128;
        let topk = 10;

        let thread_num = avail_threads().max(1);

        let input_tensor = Tensor::<f16>::from_cache(
            vec![batch_size, k],
            "model.layers.0.input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let weight_tensor = Tensor::<f16>::from_cache(
            vec![n, k],
            "weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // A = 1
        let m = batch_size;
        let mut a = vec![0.0f16; m * k];
        for x in &mut a {
            *x = 1.0f16;
        }
        unsafe {
            input_tensor
                .data
                .copy_from_nonoverlapping(a.as_ptr(), a.len());
        }

        // B_nt: bias(j) = f16(j*0.001), no ties
        let mut b_nt = vec![0.0f16; n * k];
        for j in 0..n {
            let bias_f16: f16 = (j as f32 * 0.001) as f16;
            for kk in 0..k {
                b_nt[j * k + kk] = bias_f16;
            }
        }
        unsafe {
            weight_tensor
                .data
                .copy_from_nonoverlapping(b_nt.as_ptr(), b_nt.len());
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let (indice_ptr, value_tensor) = input_tensor.matmul_local_topk(
            &weight_tensor,
            params,
            topk,
            "model.layers.0.matmul_local_topk".to_string(),
        );

        for op in operator_queue.borrow_mut().iter() {
            for tid in 0..thread_num {
                op.run(m, 1, thread_num, tid, &[], &[], &[], &mut Vec::new());
            }
        }

        let out_len = m * thread_num * topk;
        let indices = unsafe { std::slice::from_raw_parts(indice_ptr, out_len) };
        let values = unsafe { std::slice::from_raw_parts(value_tensor.data, out_len) };

        let expected_indices: Vec<usize> = (0..topk).map(|r| n - 1 - r).collect();

        let expected_value = |j: usize| -> f32 {
            let bias_f16: f16 = (j as f32 * 0.001) as f16;
            (k as f32) * (bias_f16 as f32)
        };

        for row in 0..m {
            let mut merged: Vec<(usize, f32)> = Vec::with_capacity(thread_num * topk);
            for tid in 0..thread_num {
                let base = row * (thread_num * topk) + tid * topk;
                for r in 0..topk {
                    merged.push((indices[base + r], values[base + r] as f32));
                }
            }

            merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let final_topk = &merged[..topk];

            for r in 0..topk {
                let (got_idx, got_val) = final_topk[r];
                let exp_idx = expected_indices[r];
                let exp_val = expected_value(exp_idx);

                assert_eq!(
                    got_idx, exp_idx,
                    "Index mismatch at row {}, rank {}: got {}, expected {}",
                    row, r, got_idx, exp_idx
                );

                assert!(
                    (got_val - exp_val).abs() < 0.1,
                    "Value mismatch at row {}, rank {}: got {}, expected {}",
                    row,
                    r,
                    got_val,
                    exp_val
                );
            }
        }
    }

    // ============================================================
    // MatMul / MatMulAdd — B is NT now
    // ============================================================

    #[test]
    fn test_matmul_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let batch_size = 12;
        let hidden_size = 64; // K
        let intermediate_size = 96; // N

        let input_tensor = Tensor::<f16>::from_cache(
            vec![batch_size, hidden_size],
            "model.layers.0.input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let weight_tensor = Tensor::<f16>::from_cache(
            vec![intermediate_size, hidden_size],
            "weight.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let m = batch_size;
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

        // B_nt: N×K
        let mut weight_data_nt = vec![0.0f16; n * k];
        for j in 0..n {
            for kk in 0..k {
                weight_data_nt[j * k + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.1) as f16;
            }
        }
        unsafe {
            weight_tensor
                .data
                .copy_from_nonoverlapping(weight_data_nt.as_ptr(), n * k);
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
            1,
            false,
            "model.layers.0.matmul".to_string(),
        );

        for op in operator_queue.borrow_mut().iter() {
            op.run(m, 1, 1, 0, &[], &[], &[], &mut Vec::new());
        }

        let out_len = m * n;
        let output_data = unsafe { std::slice::from_raw_parts(output_tensor.data, out_len) };

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a = input_data[i * k + kk] as f32;
                    let b = weight_data_nt[j * k + kk] as f32; // NT
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

        let batch_size = 12;
        let hidden_size = 64; // K
        let intermediate_size = 96; // N

        let input_tensor = Tensor::<f16>::from_cache(
            vec![batch_size, hidden_size],
            "model.layers.0.input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let weight_tensor = Tensor::<f16>::from_cache(
            vec![intermediate_size, hidden_size],
            "weight.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let bias_tensor = Tensor::<f16>::from_cache(
            vec![batch_size, intermediate_size],
            "bias.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let m = batch_size;
        let k = hidden_size;
        let n = intermediate_size;

        // A
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

        // B_nt
        let mut weight_data_nt = vec![0.0f16; n * k];
        for j in 0..n {
            for kk in 0..k {
                weight_data_nt[j * k + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.1) as f16;
            }
        }
        unsafe {
            weight_tensor
                .data
                .copy_from_nonoverlapping(weight_data_nt.as_ptr(), n * k);
        }

        // bias
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
            "model.layers.0.matmul_add".to_string(),
        );

        for op in operator_queue.borrow_mut().iter() {
            op.run(m, 1, 1, 0, &[], &[], &[], &mut Vec::new());
        }

        let out_len = m * n;
        let output_data = unsafe { std::slice::from_raw_parts(output_tensor.data, out_len) };

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (input_data[i * k + kk] as f32) * (weight_data_nt[j * k + kk] as f32);
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

    // ============================================================
    // ExpertsMatMulSilu / ExpertsMatMulDown / ExpertsMergeAdd
    // weights now NT: [E, I, H] and [E, H, Hmid] respectively
    // ============================================================

    /// reference: out[e,b,i] = silu(sum_k a[b,k]*w_gate_nt[e,i,k]) * (sum_k a[b,k]*w_up_nt[e,i,k])
    fn ref_experts_silu_f32(
        a: &[f16],                  // [B,H]
        w_gate_nt: &[f16],          // [E,I,H] row-major
        w_up_nt: &[f16],            // [E,I,H]
        experts_indicator: &[bool], // [E]
        indice: &[bool],            // [E,B]
        out: &mut [f32],            // [E,B,I]
        b: usize,
        h: usize,
        i: usize,
        e: usize,
    ) {
        for v in out.iter_mut() {
            *v = 0.0;
        }

        for ex in 0..e {
            if !experts_indicator[ex] {
                continue;
            }
            for bb in 0..b {
                if !indice[ex * b + bb] {
                    continue;
                }
                for ii in 0..i {
                    let mut g = 0.0f32;
                    let mut u = 0.0f32;
                    for kk in 0..h {
                        let a_v = a[bb * h + kk] as f32;
                        let wg = w_gate_nt[ex * (i * h) + ii * h + kk] as f32;
                        let wu = w_up_nt[ex * (i * h) + ii * h + kk] as f32;
                        g += a_v * wg;
                        u += a_v * wu;
                    }
                    out[ex * (b * i) + bb * i + ii] = silu_f32(g) * u;
                }
            }
        }
    }

    #[inline]
    fn slot_of(topk: &[usize], b: usize, ktop: usize, e: usize) -> usize {
        let row = &topk[b * ktop..b * ktop + ktop];
        row.iter().position(|&x| x == e).unwrap_or(0)
    }

    fn ref_down_f32(
        nonlin: &[f16],   // [E,B,Hmid]
        wdown_nt: &[f16], // [E,H,Hmid] row-major (NT)
        experts_indicator: &[bool],
        indice: &[bool],     // [E,B]
        weight: &[f16],      // [E,B]
        topk: &[usize],      // [B,Ktop]
        out_ref: &mut [f32], // [B,Ktop,H]
        e: usize,
        b: usize,
        hmid: usize,
        h: usize,
        ktop: usize,
    ) {
        for v in out_ref.iter_mut() {
            *v = 0.0;
        }

        for ex in 0..e {
            if !experts_indicator[ex] {
                continue;
            }
            for bb in 0..b {
                if !indice[ex * b + bb] {
                    continue;
                }
                let s = slot_of(topk, bb, ktop, ex);
                let w = f32_from_f16(weight[ex * b + bb]);

                for j in 0..h {
                    let mut acc = 0.0f32;
                    for kk in 0..hmid {
                        let a = f32_from_f16(nonlin[(ex * b + bb) * hmid + kk]);
                        // NT: [j * hmid + kk]
                        let bj = f32_from_f16(wdown_nt[ex * (h * hmid) + j * hmid + kk]);
                        acc += a * bj;
                    }
                    out_ref[(bb * ktop + s) * h + j] += w * acc;
                }
            }
        }
    }

    #[test]
    fn test_experts_matmul_silu_f16_tensor_api() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let batch_size = 12;
        let hidden = 64; // H
        let inter = 64; // I
        let num_experts = 2;

        let input = Tensor::<f16>::from_cache(
            vec![batch_size, hidden],
            "model.layers.0.input".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // shape is [E, I, H], raw mem_mgr also [E, I, H] row-major (NT)
        let gate_w = Tensor::<f16>::from_cache(
            vec![num_experts, inter, hidden],
            "gate.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );
        let up_w = Tensor::<f16>::from_cache(
            vec![num_experts, inter, hidden],
            "up.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let b = batch_size;

        let experts_indicator = unsafe { allocate_init(num_experts, false) };
        let indice_ptr = unsafe { allocate_init(num_experts * b, false) };

        unsafe {
            *experts_indicator.add(0) = true;
            *experts_indicator.add(1) = false;

            for bb in 0..b {
                *indice_ptr.add(0 * b + bb) = true;
                *indice_ptr.add(b + bb) = false;
            }
        }

        // input init
        let mut a = vec![0.0f16; b * hidden];
        for bb in 0..b {
            for kk in 0..hidden {
                a[bb * hidden + kk] = (((bb * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
            }
        }
        unsafe {
            input.data.copy_from_nonoverlapping(a.as_ptr(), a.len());
        }

        // weights init: [E, I, H] row-major
        let per_elems = inter * hidden;
        let mut wg_nt = vec![0.0f16; num_experts * per_elems];
        let mut wu_nt = vec![0.0f16; num_experts * per_elems];

        for e in 0..num_experts {
            for ii in 0..inter {
                for kk in 0..hidden {
                    let base_g = ((kk * 5 + ii * 11 + e * 13) % 23) as f32 * 0.01;
                    let base_u = ((kk * 9 + ii * 7 + e * 17) % 29) as f32 * 0.01;
                    wg_nt[e * per_elems + ii * hidden + kk] = base_g as f16;
                    wu_nt[e * per_elems + ii * hidden + kk] = base_u as f16;
                }
            }
        }

        unsafe {
            gate_w
                .data
                .copy_from_nonoverlapping(wg_nt.as_ptr(), wg_nt.len());
            up_w.data
                .copy_from_nonoverlapping(wu_nt.as_ptr(), wu_nt.len());
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let out = input.experts_matmul_silu_mul_matmul(
            &gate_w,
            &up_w,
            experts_indicator,
            indice_ptr,
            params,
            false,
            "model.layers.0.experts_silu".to_string(),
        );

        assert_eq!(out.shape, vec![num_experts, batch_size, inter]);
        assert_eq!(operator_queue.borrow().len(), 1);
        assert!(matches!(
            &operator_queue.borrow()[0],
            Operator::ExpertsMatMulSilu(_)
        ));

        let thread_num = avail_threads();

        for op in operator_queue.borrow_mut().iter() {
            for tid in 0..thread_num {
                op.run(b, 1, thread_num, tid, &[], &[], &[], &mut Vec::new());
            }
        }

        let out_len = num_experts * b * inter;
        let out_got = unsafe { std::slice::from_raw_parts(out.data, out_len) };

        // reference for expert0
        for bb in 0..b {
            for ii in 0..inter {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for kk in 0..hidden {
                    let a_v = a[bb * hidden + kk] as f32;
                    let wg_v = wg_nt[0 * per_elems + ii * hidden + kk] as f32;
                    let wu_v = wu_nt[0 * per_elems + ii * hidden + kk] as f32;
                    g += a_v * wg_v;
                    u += a_v * wu_v;
                }
                let exp = silu_f32(g) * u;

                let got = out_got[0 * (b * inter) + bb * inter + ii] as f32;
                assert!(
                    (got - exp).abs() < 0.5,
                    "Mismatch expert0 bb {} ii {}: got {}, expected {}",
                    bb,
                    ii,
                    got,
                    exp
                );
            }
        }

        // expert1 inactive -> 0
        for bb in 0..b {
            for ii in 0..inter {
                let got = out_got[(b * inter) + bb * inter + ii] as f32;
                assert!(
                    got.abs() < 1e-3,
                    "Inactive expert1 should be ~0, but got {} at bb {} ii {}",
                    got,
                    bb,
                    ii
                );
            }
        }
    }

    #[test]
    fn test_experts_matmul_down_f16_tensor_api() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let batch_size = 12;
        let num_experts = 2;

        let inter = 64; // K (KC=64)
        let hidden = 32; // N (NR=32)
        let num_experts_per_tok = 1;

        let b = batch_size;

        // input to down: [E, seq, batch, inter]
        let x = Tensor::<f16>::from_cache(
            vec![num_experts, batch_size, inter],
            "model.layers.0.experts.silu_out".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // down weights: shape [E, hidden, inter]
        // ✅ NEW contract: B is already NT (N×K) row-major in mem_mgr per expert:
        // w_nt[j * inter + kk]
        let down_w = Tensor::<f16>::from_cache(
            vec![num_experts, hidden, inter],
            "model.layers.0.down.weight".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let experts_indicator = unsafe { allocate_init(num_experts, false) };
        let indice_ptr = unsafe { allocate_init(num_experts * b, false) };
        let weight_ptr = unsafe { allocate_init(num_experts * b, 0.0f16) };
        let topk_indices_ptr = unsafe { allocate_init(b * num_experts_per_tok, 0usize) };

        unsafe {
            *experts_indicator.add(0) = true;
            *experts_indicator.add(1) = false;

            for t in 0..b {
                *indice_ptr.add(0 * b + t) = true;
                *indice_ptr.add(b + t) = false;

                *weight_ptr.add(0 * b + t) = 1.0f16;
                *weight_ptr.add(b + t) = 0.0f16;

                *topk_indices_ptr.add(t) = 0usize;
            }
        }

        // init X: expert0 pattern, expert1 zeros
        let mut x_e0 = vec![0.0f16; b * inter];
        for t in 0..b {
            for kk in 0..inter {
                x_e0[t * inter + kk] = (((t * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
            }
        }
        unsafe {
            x.data
                .add(0 * (b * inter))
                .copy_from_nonoverlapping(x_e0.as_ptr(), x_e0.len());
            for i in 0..(b * inter) {
                *x.data.add((b * inter) + i) = 0.0f16;
            }
        }

        // init W_down:
        // ✅ NEW: per expert is NT (N×K) = hidden × inter row-major:
        // w_e0[j * inter + kk]
        let per_w = inter * hidden;
        let mut w_e0 = vec![0.0f16; per_w];
        let mut w_e1 = vec![0.0f16; per_w];

        for j in 0..hidden {
            for kk in 0..inter {
                // 跟以前一样的 deterministic pattern，只是存储索引变了
                w_e0[j * inter + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.01) as f16;
                w_e1[j * inter + kk] = (((kk * 3 + j * 7) % 29) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            down_w
                .data
                .add(0 * per_w)
                .copy_from_nonoverlapping(w_e0.as_ptr(), per_w);
            down_w
                .data
                .add(per_w)
                .copy_from_nonoverlapping(w_e1.as_ptr(), per_w);
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let out = x.experts_matmul_mul(
            &down_w,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            topk_indices_ptr,
            num_experts_per_tok,
            params,
            false,
            "model.layers.0.experts_down".to_string(),
        );

        assert_eq!(out.shape, vec![batch_size, num_experts_per_tok, hidden]);
        assert_eq!(operator_queue.borrow().len(), 1);
        assert!(matches!(
            &operator_queue.borrow()[0],
            Operator::ExpertsMatMulDown(_)
        ));

        // ✅ IMPORTANT: down does out += acc * factor, so zero out first
        let out_len = b * num_experts_per_tok * hidden;
        unsafe {
            for i in 0..out_len {
                *out.data.add(i) = 0.0f16;
            }
        }

        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        for op in operator_queue.borrow_mut().iter() {
            for tid in 0..thread_num {
                op.run(b, 1, thread_num, tid, &[], &[], &[], &mut Vec::new());
            }
        }

        // verify reference:
        // out[t, 0, j] = sum_k x_e0[t,k] * w_e0_nt[j,k]
        let out_got = unsafe { std::slice::from_raw_parts(out.data, out_len) };
        for t in 0..b {
            for j in 0..hidden {
                let mut sum = 0.0f32;
                for kk in 0..inter {
                    let x_v = x_e0[t * inter + kk] as f32;
                    let w_v = w_e0[j * inter + kk] as f32; // ✅ NT indexing
                    sum += x_v * w_v;
                }
                let got = out_got[t * hidden + j] as f32;
                assert!(
                    (got - sum).abs() < 0.5,
                    "Down mismatch token {} col {}: got {}, expected {}",
                    t,
                    j,
                    got,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_experts_merge_add_f16_tensor_api_k2_slot1_zero() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
        let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

        let batch_size = 12;
        let num_tokens = batch_size;

        let num_experts = 2; // 仅用于 reset gating（我们这里 reset_gating=false）
        let k = 2usize; // num_experts_per_token == K
        let hidden = 64usize;

        // input ptr layout: [num_tokens, K, H]
        let input = Tensor::<f16>::from_cache(
            vec![batch_size, k, hidden],
            "model.layers.0.moe.down_out".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let residual = Tensor::<f16>::from_cache(
            vec![batch_size, hidden],
            "model.layers.0.residual".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        // routing buffers（reset_gating=false 不会用来选择，只会在 reset_gating=true 时清零）
        let experts_indicator = unsafe { allocate_init(num_experts, false) };
        let indice_ptr = unsafe { allocate_init(num_experts * num_tokens, false) };

        unsafe {
            *experts_indicator.add(0) = true;
            *experts_indicator.add(1) = true;
            for t in 0..num_tokens {
                *indice_ptr.add(0 * num_tokens + t) = true;
                *indice_ptr.add(num_tokens + t) = true;
            }
        }

        // init residual
        let mut r = vec![0.0f16; num_tokens * hidden];
        for t in 0..num_tokens {
            for h in 0..hidden {
                r[t * hidden + h] = (((t * 2 + h * 5) % 17) as f32 * 0.01) as f16;
            }
        }
        unsafe {
            residual.data.copy_from_nonoverlapping(r.as_ptr(), r.len());
        }

        // init input: slot0 = pattern, slot1 = 0
        let mut slot0 = vec![0.0f16; num_tokens * hidden];
        for t in 0..num_tokens {
            for h in 0..hidden {
                slot0[t * hidden + h] = (((t * 7 + h * 3) % 19) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            // 全清零
            let total = num_tokens * k * hidden;
            for i in 0..total {
                *input.data.add(i) = 0.0f16;
            }
            // 写 slot0
            for t in 0..num_tokens {
                let base = t * (k * hidden);
                input
                    .data
                    .add(base + 0 * hidden)
                    .copy_from_nonoverlapping(slot0.as_ptr().add(t * hidden), hidden);
            }
        }

        // build via Tensor API
        let out = input.experts_merge_add(
            &residual,
            experts_indicator,
            indice_ptr,
            num_experts,
            false,
            "model.layers.0.experts_merge_add".to_string(),
        );

        assert_eq!(out.shape, vec![batch_size, hidden]);
        assert_eq!(operator_queue.borrow().len(), 1);
        assert!(matches!(
            &operator_queue.borrow()[0],
            Operator::ExpertsMergeAdd(_)
        ));

        // run
        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        for op in operator_queue.borrow_mut().iter() {
            for tid in 0..thread_num {
                op.run(
                    num_tokens,
                    1,
                    thread_num,
                    tid,
                    &[],
                    &[],
                    &[],
                    &mut Vec::new(),
                );
            }
        }

        // verify: out = residual + slot0 + slot1(0)
        let out_len = num_tokens * hidden;
        let out_got = unsafe { std::slice::from_raw_parts(out.data, out_len) };

        for t in 0..num_tokens {
            for h in 0..hidden {
                let exp = (r[t * hidden + h] as f32) + (slot0[t * hidden + h] as f32);
                let got = out_got[t * hidden + h] as f32;
                assert!(
                    (got - exp).abs() < 1e-3,
                    "MergeAdd mismatch token {} h {}: got {}, expected {}",
                    t,
                    h,
                    got,
                    exp
                );
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f16;

    fn avail_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    #[inline]
    fn skip_if_no_avx512fp16() {
        if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported on this machine, skipping test.");
        }
    }

    #[test]
    fn test_matmul_new_uses_bnt_directly_f16() {
        if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        const K: usize = 8;
        const N: usize = 6;
        const M: usize = 3;

        // ✅ NEW contract: B is already NT (N×K) row-major
        let mut a = vec![0.0f16; M * K];
        let mut b_nt = vec![0.0f16; N * K];
        let mut c = vec![0.0f16; M * N];

        // b_nt 用唯一值：b_nt[j*K + kk] = 100*j + kk
        for j in 0..N {
            for kk in 0..K {
                let v = (100 * j + kk) as f32;
                b_nt[j * K + kk] = v as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 4, // kc
            a_row_step_micro: 3,
            b_row_step_micro: 2, // nr
        };

        let matmul = unsafe {
            MatMul::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                c.as_mut_ptr(),
                false,
                params,
                M,
                N,
                K,
                false,
            )
        };

        // 只验证：内部使用的 B 视图（ptr2.ptr）在“语义上”保持 b_nt 的 N×K 索引一致
        // 注意：这里不要求 ptr2.ptr 与输入同一地址（实现可能做 pack/copy）
        let internal_b_nt = unsafe { std::slice::from_raw_parts(matmul.ptr2.ptr, N * K) };
        for j in 0..N {
            for kk in 0..K {
                let got = internal_b_nt[j * K + kk] as f32;
                let expected = b_nt[j * K + kk] as f32;
                assert_abs_diff_eq!(got, expected, epsilon = 0.0);
            }
        }
    }

    #[test]
    fn test_matmul_panel_pool_non_overlapping_f16() {
        if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        const M: usize = 3;
        const K: usize = 64;
        const N: usize = 32;

        let threads = avail_threads().min(16);

        let a = vec![0.0f16; M * K];
        // ✅ NEW contract: B is N×K (still same element count as K×N here, but indexing differs)
        let b_nt = vec![0.0f16; N * K];
        let mut c = vec![0.0f16; M * N];

        let params = MatMulParams {
            a_row_step_macro: 3,
            b_row_step_macro: 32,
            column_step_macro: 64, // kc
            a_row_step_micro: 3,
            b_row_step_micro: 32, // nr
        };

        let matmul = unsafe {
            MatMul::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                c.as_mut_ptr(),
                false,
                params,
                M,
                N,
                K,
                false,
            )
        };

        assert!(matmul.panel_threads() >= 1);

        let kc = matmul.params.column_step_macro.max(1);
        let nr = matmul.params.b_row_step_micro.max(1);
        let stride = kc * nr;

        let used = threads.min(matmul.panel_threads());
        for tid in 0..used {
            let p = matmul.thread_b_panel_ptr(tid);
            unsafe {
                for i in 0..stride {
                    *p.add(i) = ((tid + 1) as f32) as f16;
                }
            }
        }

        for tid in 0..used {
            let p = matmul.thread_b_panel_ptr(tid);
            let expected = ((tid + 1) as f32) as f16;
            unsafe {
                for i in 0..stride {
                    let got = *p.add(i);
                    assert_eq!(
                        got, expected,
                        "panel overlap detected at tid={}, i={}",
                        tid, i
                    );
                }
            }
        }
    }

    #[test]
    fn test_matmul_runner_f16_multi_tile_and_threads() {
        if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        const M: usize = 12;
        const K: usize = 64;
        const N: usize = 128;

        let thread_num = avail_threads().min(16);

        let mut a = vec![0.0f16; M * K];
        // ✅ NEW: B is already NT: N×K row-major
        let mut b_nt = vec![0.0f16; N * K];
        let mut c = vec![0.0f16; M * N];

        for i in 0..M {
            for kk in 0..K {
                a[i * K + kk] = (((i * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
            }
        }
        for j in 0..N {
            for kk in 0..K {
                // 同 pattern，只是按 NT 写
                b_nt[j * K + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.01) as f16;
            }
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 128,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let matmul = unsafe {
            MatMul::<f16>::new(
                a.as_ptr(),
                b_nt.as_ptr(),
                c.as_mut_ptr(),
                false,
                params,
                M,
                N,
                K,
                false,
            )
        };

        let cpu_num = thread_num.min(matmul.panel_threads()).max(1);

        for tid in 0..cpu_num {
            matmul.run(M, 0, cpu_num, tid);
        }

        // reference: sum += A[i,kk] * B_nt[j,kk]
        for i in 0..M {
            for j in 0..N {
                let mut sum = 0.0f32;
                for kk in 0..K {
                    sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
                }
                let got = c[i * N + j] as f32;
                assert_abs_diff_eq!(got, sum, epsilon = 5e-1);
            }
        }
    }

    // 你原来的两个 test 保持不动即可（我就不贴了）
}
// 你原来的两个 test 保持不动即可（我就不贴了）

/*
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
        let output_shape = vec![self.shape[0], self.shape[1], b_tensor.shape[2]];
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
            // self.shape[0],
            self.shape[1],
            self.shape[2],
            output_to_kv,
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
            // self.shape[0],
            self.shape[1],
            self.shape[2],
        ));
        self.operator_queue.borrow_mut().push(operator);
        output_tensor
    }


*/
