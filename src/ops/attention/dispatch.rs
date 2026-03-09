use std::f16;
use std::ops::{Add, Div, Mul, Sub};

use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::kernel;
use crate::ops::traits::AttentionTrait;

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
        kernel::scalar::flash_attention::flash_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            self.inverse_sqrt_head,
            self.head_size,
            self.head_size,
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
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::flash_attention::flash_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            self.inverse_sqrt_head,
            self.head_size,
            self.head_size,
            position,
        );

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::scalar::flash_attention::flash_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            self.inverse_sqrt_head,
            self.head_size,
            self.head_size,
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
        kernel::scalar::flash_attention::flash_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            self.inverse_sqrt_head,
            self.head_size,
            self.head_size,
            position,
        );
    }
}
