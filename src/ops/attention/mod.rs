mod dispatch;
mod split_sequence;

use std::ops::{Add, Div, Mul, Sub};

use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::common::sequence_slice::SequenceSlice;
use split_sequence::split_sequence_by_triangle;

#[derive(Clone)]
pub struct Attention<T> {
    q_ptr: ConstPtr<T>,
    k_ptr: ConstPtr<T>,
    v_ptr: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    batch_size: usize,
    attention_head_num: usize,
    kv_head_num: usize,
    seq_len: usize,
    col_size: usize,
    head_size: usize,
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
        seq_len: usize,
        col_size: usize,
        head_size: usize,
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
            seq_len: seq_len,
            col_size: col_size,
            head_size: head_size,
            inverse_sqrt_head: inverse_sqrt_head,
            decode_only_flag: decode_only_flag,
        }
    }

    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        attention_list: &[SequenceSlice],
        thread_num: usize,
        thread_id: usize,
    ) {
        unsafe {
            let q_ptr = self.q_ptr.ptr;
            let k_ptr = self.k_ptr.ptr;
            let v_ptr = self.v_ptr.ptr;
            let output_ptr = self.output_ptr.ptr;
            let q_stride = self.attention_head_num * self.head_size;
            let kv_batch_stride = self.kv_head_num * self.seq_len * self.head_size;
            let group_num = self.attention_head_num / self.kv_head_num;
            let kv_head_stride = self.seq_len * self.head_size;

            for slice in attention_list {
                let batch_index = slice.batch_index;
                if batch_index >= self.batch_size {
                    continue;
                }

                let Some((begin, end)) =
                    split_sequence_by_triangle(slice.length, thread_num, thread_id)
                else {
                    continue;
                };

                let k_batch_ptr = k_ptr.add(batch_index * kv_batch_stride);
                let v_batch_ptr = v_ptr.add(batch_index * kv_batch_stride);
                let token_start_index = slice.token_start_index;
                let position_start_index = slice.sequence_index;

                for kv_head in 0..self.kv_head_num {
                    let q_head_offset = kv_head * group_num * self.head_size;
                    let kv_head_offset = kv_head * kv_head_stride;
                    let _k_head_ptr = k_batch_ptr.add(kv_head_offset);
                    let _v_head_ptr = v_batch_ptr.add(kv_head_offset);

                    for chunk_begin in (begin..end).step_by(self.col_size) {
                        let chunk_end = (chunk_begin + self.col_size).min(end);
                        let block_token_begin = token_start_index + chunk_begin;
                        let block_token_end = token_start_index + chunk_end;
                        let block_position_begin = position_start_index + chunk_begin;
                        let block_position_end = position_start_index + chunk_end;
                        let _q_block_ptr = q_ptr.add(block_token_begin * q_stride + q_head_offset);
                        let _output_block_ptr =
                            output_ptr.add(block_token_begin * q_stride + q_head_offset);

                        let _ = (block_token_end, block_position_begin, block_position_end);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;
}
