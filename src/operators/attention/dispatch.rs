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
        row_begin: usize,
        row_end: usize,
        col_begin: usize,
        col_end: usize,
        total_col_end: usize,
        sequence_index: usize,
        running_max: &mut [T],
        running_denom: &mut [T],
        scores: &mut [T],
    ) {
        kernel::scalar::block_flash_attention::block_flash_attention(
            q_ptr,
            output_ptr,
            row_begin,
            row_end,
            col_begin,
            col_end,
            total_col_end,
            k_ptr,
            v_ptr,
            self.head_size,
            self.inverse_sqrt_head,
            sequence_index,
            running_max,
            running_denom,
            scores,
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
        row_begin: usize,
        row_end: usize,
        col_begin: usize,
        col_end: usize,
        total_col_end: usize,
        sequence_index: usize,
        running_max: &mut [f16],
        running_denom: &mut [f16],
        scores: &mut [f16],
    ) {
        kernel::scalar::block_flash_attention::block_flash_attention(
            q_ptr,
            output_ptr,
            row_begin,
            row_end,
            col_begin,
            col_end,
            total_col_end,
            k_ptr,
            v_ptr,
            self.head_size,
            self.inverse_sqrt_head,
            sequence_index,
            running_max,
            running_denom,
            scores,
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
        row_begin: usize,
        row_end: usize,
        col_begin: usize,
        col_end: usize,
        total_col_end: usize,
        sequence_index: usize,
        running_max: &mut [f32],
        running_denom: &mut [f32],
        scores: &mut [f32],
    ) {
        kernel::scalar::block_flash_attention::block_flash_attention(
            q_ptr,
            output_ptr,
            row_begin,
            row_end,
            col_begin,
            col_end,
            total_col_end,
            k_ptr,
            v_ptr,
            self.head_size,
            self.inverse_sqrt_head,
            sequence_index,
            running_max,
            running_denom,
            scores,
        );
    }
}
