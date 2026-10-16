use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, FromNumber, NegInfinity, Sigmoid, Sqrt};
use crate::operators::movement::{HeadLookup, LiftVector};
use crate::operators::operator::Operator;
use crate::operators::routing::SampleTokens;
use crate::operators::transform::{
    AddRMSZipMap, AddZipMap, LookupRMSMap, RMSMap, SigmoidMap, SiluMulZipMap,
};

use super::{GlobalOperatorQueue, Tensor};

impl<T> Tensor<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Add<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid
        + Sqrt
        + AddAssign
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    pub fn add(&self, b_tensor: &Tensor<T>, decode_only_flag: bool, tensor_name: String) -> Self {
        let output_tensor = Self::from_mem_pool(self.shape.clone(), tensor_name);
        let (head_num, head_size) = match self.shape.as_slice() {
            [_, cols] => (1, *cols),
            [_, heads, size] => (*heads, *size),
            _ => (self.last_dim(), 1),
        };
        let operator = Operator::AddZipMap(AddZipMap::new(
            self.data,
            b_tensor.data,
            output_tensor.data,
            head_num,
            head_size,
            decode_only_flag,
        ));
        Self::enqueue(operator);
        output_tensor
    }

    pub fn silu_mul(
        &self,
        b_tensor: &Tensor<T>,
        decode_only_flag: bool,
        tensor_name: String,
    ) -> Self {
        let output_tensor = Self::from_mem_pool(self.shape.clone(), tensor_name);
        let (head_num, head_size) = match self.shape.as_slice() {
            [_, cols] => (1, *cols),
            [_, heads, size] => (*heads, *size),
            _ => (self.last_dim(), 1),
        };
        let operator = Operator::SiluMulZipMap(SiluMulZipMap::new_with_decode_only(
            self.data,
            b_tensor.data,
            output_tensor.data,
            head_num,
            head_size,
            decode_only_flag,
        ));
        Self::enqueue(operator);
        output_tensor
    }

    pub fn add_rms(
        &self,
        b_tensor: &Tensor<T>,
        _weight: *const T,
        eps: T,
        tensor_name: String,
    ) -> Self {
        let output_tensor = Self::from_mem_pool(self.shape.clone(), tensor_name);

        let operator = Operator::AddRMSZipMap(AddRMSZipMap::new(
            self.data,
            b_tensor.data,
            output_tensor.data,
            self.shape[1],
            eps,
        ));
        Self::enqueue(operator);
        output_tensor
    }

    pub fn sigmoid(&self, tensor_name: String) -> Self {
        let output_tensor = Self::from_mem_pool(self.shape.clone(), tensor_name);
        let operator = Operator::SigmoidMap(SigmoidMap::new(
            self.data,
            output_tensor.data,
            self.element_count(),
        ));
        Self::enqueue(operator);
        output_tensor
    }

    pub fn lookup(
        head_token_ptr: *const usize,
        word_embedding: &Tensor<T>,
        token_capacity: usize,
        scope_name: String,
    ) -> Self {
        let output_tensor = Self::from_mem_pool(
            vec![token_capacity, word_embedding.shape[1]],
            format!("{}.output", scope_name),
        );

        let operator = Operator::HeadLookup(HeadLookup::new(
            head_token_ptr,
            word_embedding.data,
            output_tensor.data,
            word_embedding.shape[1],
        ));

        Self::enqueue(operator);
        output_tensor
    }

    pub fn lookup_rms(
        sequences_ptr: *const usize,
        word_embedding: &Tensor<T>,
        norm_weight: &Tensor<T>,
        token_capacity: usize,
        sequence_stride: usize,
        eps: T,
        scope_name: String,
    ) -> (Self, Self) {
        let output_hidden_tensor = Self::from_mem_pool(
            vec![token_capacity, word_embedding.shape[1]],
            format!("{}.output_hidden", scope_name),
        );

        let output_normal_tensor = Self::from_mem_pool(
            vec![token_capacity, word_embedding.shape[1]],
            format!("{}.output_normal", scope_name),
        );

        let operator = Operator::LookupRMSMap(LookupRMSMap::new(
            sequences_ptr,
            word_embedding.data,
            norm_weight.data,
            output_hidden_tensor.data,
            output_normal_tensor.data,
            sequence_stride,
            word_embedding.shape[1],
            eps,
        ));

        Self::enqueue(operator);
        (output_hidden_tensor, output_normal_tensor)
    }

    pub fn lift_vector(&self) {
        let row_len = self.shape.iter().skip(1).product();
        let operator = Operator::LiftVector(LiftVector::new(self.data, row_len));
        Self::enqueue(operator);
    }

    pub fn rms(
        &self,
        weight: &Tensor<T>,
        eps: T,
        decode_only_flag: bool,
        scope_name: String,
    ) -> Self {
        let output_tensor = Self::output_tensor(self.shape.clone(), &scope_name);

        let operator = Operator::RMSMap(RMSMap::new(
            self.data,
            weight.data,
            output_tensor.data,
            self.last_dim(),
            eps,
            decode_only_flag,
        ));
        Self::enqueue(operator);
        output_tensor
    }
}

// Tensor builder for the already-registered `SampleTokens` operator (not a new
// compute operator). `SampleTokens` needs `FromNumber` on top of the arithmetic
// bounds the other `ops.rs` builders use, so it lives in its own `impl` block to
// leave the main block untouched.
impl<T> Tensor<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Add<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid
        + Sqrt
        + AddAssign
        + FromNumber
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    /// Rewrite the logits row in place into a sampled distribution and write the
    /// sampled token id into `output_sequences[batch_index * sequence_stride + ..]`.
    pub fn sample_tokens(
        &self,
        output_sequences: *mut usize,
        batch_temperature: *mut T,
        sequence_stride: usize,
        _tensor_name: String,
    ) -> Self {
        let operator = Operator::SampleTokens(SampleTokens::new(
            self.data,
            output_sequences,
            batch_temperature,
            sequence_stride,
            self.last_dim(),
        ));
        Self::enqueue(operator);
        self.clone()
    }
}
