use std::f16;
use std::ops::{Add, Div, Mul, Sub};

use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::kernel;
use crate::operators::traits::AttentionTrait;

use super::Attention;

impl<T> AttentionTrait<T> for Attention<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + NegInfinity
        + Exp,
{
    default fn compute(
        &self,
        q_ptr: *const T,
        k_ptr: *const T,
        v_ptr: *const T,
        output_ptr: *mut T,
        position: usize,
    ) {
        kernel::scalar::block_flash_attention::block_flash_attention(
            q_ptr,
            output_ptr,
            0,
            1,
            position + 1,
            k_ptr,
            v_ptr,
            self.head_size,
            self.col_size,
            self.inverse_sqrt_head,
            position,
        );
    }
}

impl AttentionTrait<f16> for Attention<f16> {
    fn compute(
        &self,
        q_ptr: *const f16,
        k_ptr: *const f16,
        v_ptr: *const f16,
        output_ptr: *mut f16,
        position: usize,
    ) {
        kernel::scalar::block_flash_attention::block_flash_attention(
            q_ptr,
            output_ptr,
            0,
            1,
            position + 1,
            k_ptr,
            v_ptr,
            self.head_size,
            self.col_size,
            self.inverse_sqrt_head,
            position,
        );
    }
}

impl AttentionTrait<f32> for Attention<f32> {
    fn compute(
        &self,
        q_ptr: *const f32,
        k_ptr: *const f32,
        v_ptr: *const f32,
        output_ptr: *mut f32,
        position: usize,
    ) {
        kernel::scalar::block_flash_attention::block_flash_attention(
            q_ptr,
            output_ptr,
            0,
            1,
            position + 1,
            k_ptr,
            v_ptr,
            self.head_size,
            self.col_size,
            self.inverse_sqrt_head,
            position,
        );
    }
}
