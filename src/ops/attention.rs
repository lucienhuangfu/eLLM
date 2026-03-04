use std::f16;
use std::ops::{Add, Div, Mul, Sub};

use crate::common::sequence_slice::SequenceSlice;
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::kernel;
use crate::ops::traits::AttentionTrait;

#[derive(Clone)]
pub struct Attention<T> {
    q_ptr: ConstPtr<T>,
    k_ptr: ConstPtr<T>,
    v_ptr: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    batch_size: usize,
    attention_head_num: usize,
    kv_head_num: usize,
    head_size: usize,
    kv_strides: Vec<usize>,
    inverse_sqrt_head: T,
    decode_only_flag: bool,
}

impl<T> Attention<T>
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
    pub fn new(
        // [batch_size, seq_len, kv_head_num, group_num, head_dim]
        q_ptr: *const T,
        // [batch_size, kv_head_num, seq_len, head_dim]
        k_ptr: *const T,
        v_ptr: *const T,
        output_ptr: *mut T,
        batch_size: usize,
        attention_head_num: usize,
        kv_head_num: usize,
        head_size: usize,
        kv_strides: Vec<usize>,
        inverse_sqrt_head: T,
        decode_only_flag: bool,
    ) -> Self {
        Self {
            q_ptr: ConstPtr { ptr: q_ptr },
            k_ptr: ConstPtr { ptr: k_ptr },
            v_ptr: ConstPtr { ptr: v_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            batch_size: batch_size,
            attention_head_num: attention_head_num,
            kv_head_num: kv_head_num,
            head_size: head_size,
            kv_strides: kv_strides,
            inverse_sqrt_head: inverse_sqrt_head,
            decode_only_flag: decode_only_flag,
        }
    }

    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        _thread_num: usize,
        _thread_id: usize,
        attention_list: &[SequenceSlice],
    ) {
        unsafe {
            let q_ptr = self.q_ptr.ptr;
            let k_ptr = self.k_ptr.ptr;
            let v_ptr = self.v_ptr.ptr;
            let output_ptr = self.output_ptr.ptr;
            let q_stride = self.attention_head_num * self.head_size;
            let kv_batch_stride = self.kv_strides[1];

            for slice in attention_list {
                let batch_index = slice.batch_index;
                let token_start_index = slice.token_start_index;
                let position_start_index = slice.sequence_index;

                let k_batch_ptr = k_ptr.add(batch_index * kv_batch_stride);
                let v_batch_ptr = v_ptr.add(batch_index * kv_batch_stride);

                for t in 0..slice.length {
                    let token_index = token_start_index + t;
                    let position_index = position_start_index + t;
                    let q_offset = token_index * q_stride;

                    self.compute(
                        q_ptr.add(q_offset),
                        k_batch_ptr,
                        v_batch_ptr,
                        output_ptr.add(q_offset),
                        position_index,
                    );
                }
            }
        }
    }
}

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
            self.kv_strides[2],
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
            self.kv_strides[2],
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
            self.kv_strides[2],
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
            self.kv_strides[2],
            position,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;
}
