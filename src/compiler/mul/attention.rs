use std::f16;
use std::ops::{Add, Div, Mul, Sub};
use std::ptr;

use super::super::super::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::super::super::kernel;
use crate::init::record::TokenList;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::mul_trait::AttentionTrait;
use crate::compiler::assign::assign;

#[derive(Clone)]
pub struct Attention<T> {
    q_ptr: ConstPtr<T>,
    k_ptr: ConstPtr<T>,
    v_ptr: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    token_list_ptr: ConstPtr<TokenList>,
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
        q_ptr: *const T,
        k_ptr: *const T,
        v_ptr: *const T,
        output_ptr: *mut T,
        token_list_ptr: *const TokenList,
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
            token_list_ptr: ConstPtr {
                ptr: token_list_ptr,
            },
            batch_size: batch_size,
            attention_head_num: attention_head_num,
            kv_head_num: kv_head_num,
            head_size: head_size,
            kv_strides: kv_strides,
            inverse_sqrt_head: inverse_sqrt_head,
            decode_only_flag: decode_only_flag,
        }
    }

    pub fn run(&self, prefill_size: usize, decode_size: usize, thread_num: usize, thread_id: usize) {
        if let Some((begin, end)) = assign(prefill_size, thread_num, thread_id) {
            unsafe {
                let q_ptr = self.q_ptr.ptr;
                let k_ptr = self.k_ptr.ptr;
                let v_ptr = self.v_ptr.ptr;
                let output_ptr = self.output_ptr.ptr;
                let token_records_ptr = (*self.token_list_ptr.ptr).token_records.as_ptr();
                let lift_records_ptr = (*self.token_list_ptr.ptr).lift_records.as_ptr();
                let lift_size = (*self.token_list_ptr.ptr).current_lift_size;
                let decode_end_index = prefill_size - lift_size;

                for i in begin..end {
                    let (batch_index, position_index) = if i < decode_end_index {
                        let record = &*token_records_ptr.add(i);
                        (record.batch_index, record.position_index)
                    } else {
                        let lift_offset = i - decode_end_index;
                        let lift_record = &*lift_records_ptr.add(lift_offset);
                        let batch_idx = lift_record.prefill_end_index;
                        let pos_idx = (*token_records_ptr.add(batch_idx)).position_index;
                        (batch_idx, pos_idx)
                    };

                    let q_offset = i * self.attention_head_num * self.head_size;
                    let out_offset = i * self.attention_head_num * self.head_size;

                    // K/V cache offset
                    // Assuming kv_strides[1] is batch stride.
                    let k_offset = batch_index * self.kv_strides[1];
                    let v_offset = batch_index * self.kv_strides[1];

                    self.compute(
                        q_ptr.add(q_offset),
                        k_ptr.add(k_offset),
                        v_ptr.add(v_offset),
                        // ptr::null(),
                        output_ptr.add(out_offset),
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
        // residual_ptr: *const T,
        output_ptr: *mut T,
        position: usize,
    ) {
        kernel::generic::flash_attention::flash_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            // residual_ptr,
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
        // residual_ptr: // *const f16,
        output_ptr: *mut f16,
        position: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::flash_attention::flash_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            // residual_ptr,
            output_ptr,
            self.inverse_sqrt_head,
            self.head_size,
            self.kv_strides[2],
            position,
        );

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::flash_attention::flash_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            // residual_ptr,
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
        // residual_ptr: // *const f32,
        output_ptr: *mut f32,
        position: usize,
    ) {
        kernel::generic::flash_attention::flash_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            // residual_ptr,
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
    // use super::super::chunk_attention::chunk_attention;
    use super::*;
    // use crate::ptensor::tensor_utils::get_strides;
    use approx::assert_ulps_eq;

    /*
    #[test]
    fn test_attention_mul() {
        let sequence_chunk_size = 256;
        let sequence_length = 256;
        let batch_size = 16;
        let attention_head_num = 8;
        let kv_head_num = 8;
        let head_size = 128;

        let q_shape = vec![sequence_chunk_size, batch_size, attention_head_num, head_size];
        let q_size = q_shape.iter().product();
        let q_data = vec![1.0; q_size];
        let q_strides = get_strides(&q_shape);

        let k_shape = vec![sequence_length, batch_size, kv_head_num, head_size];
        let k_strides = get_strides(&k_shape);
        let k_size = k_shape.iter().product();
        let k_data = vec![1.0; k_size];

        let v_shape = vec![sequence_length, batch_size, kv_head_num, head_size];
        let v_strides = get_strides(&v_shape);
        let v_size = v_shape.iter().product();
        let v_data = vec![1.0; v_size];

        let kv_strides2 = vec![k_strides[1], k_strides[2], k_strides[0], k_strides[3]];

        let o_shape = vec![sequence_chunk_size, batch_size, attention_head_num, head_size];
        let o_size = o_shape.iter().product();
        let mut o_data = vec![1.0; o_size];
        let o_strides = get_strides(&o_shape);



        let shape4 = vec![sequence_chunk_size, batch_size, attention_head_num, head_size];
        let size4 = shape4.iter().product();
        let mut data4 = vec![0.0; size4];
        let strides4 = get_strides(&shape4);

        let result = vec![1.0; size4];

        let thread_num: usize = num_cpus::get();
        let mut operator =
            AttentionMul::<f32>::new(
                q_data.as_ptr(),
                k_data.as_ptr(),
                v_data.as_ptr(),
                o_data.as_mut_ptr(),
                batch_size,
                attention_head_num,
                kv_head_num,
                head_size,
                kv_strides2,
                1.0,
            );

        let position_index = 0;
        let position_interval = 1;

        for i in 0..thread_num {
            // println!("{}", i);
            operator.run(position_index, position_interval, batch_size, thread_num, i);
        }

        // assert_ulps_eq!(data4[..], result[..], max_ulps = 4);
    }*/
}

