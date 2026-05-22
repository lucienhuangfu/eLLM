use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::common::num_traits::{Exp, NegInfinity, Sigmoid, Sqrt};
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::movement::LiftVector;
use crate::operators::operator::Operator;
use crate::operators::transform::{AddRMSZipMap, AddZipMap, LookupRMSMap, RMSMap, SigmoidMap};

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
        let operator = Operator::AddZipMap(AddZipMap::new(
            self.data,
            b_tensor.data,
            output_tensor.data,
            self.shape[1],
            self.shape[2],
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

    pub fn lookup_rms(
        sequences_ptr: *const usize,
        word_embedding: &Tensor<T>,
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

    pub fn rms(&self, eps: T, decode_only_flag: bool, scope_name: String) -> Self {
        let output_tensor = Self::output_tensor(self.shape.clone(), &scope_name);

        let operator = Operator::RMSMap(RMSMap::new(
            self.data,
            output_tensor.data,
            // self.shape[0],
            self.shape[1],
            eps,
            decode_only_flag,
        ));
        Self::enqueue(operator);
        output_tensor
    }
}
