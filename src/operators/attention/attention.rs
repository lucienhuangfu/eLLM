use std::ops::{Add, Div, Mul, Sub};

use super::split_sequence::{
    should_split_by_attention_head, split_attention_heads, split_sequence_by_triangle,
};
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::common::sequence_slice::SequenceSlice;

#[derive(Clone)]
pub struct Attention<T> {
    pub(super) q_ptr: ConstPtr<T>,
    pub(super) k_ptr: ConstPtr<T>,
    pub(super) v_ptr: ConstPtr<T>,
    pub(super) output_ptr: MutPtr<T>,
    pub(super) batch_size: usize,
    pub(super) attention_head_num: usize,
    pub(super) kv_head_num: usize,
    pub(super) seq_len: usize,
    pub(super) row_size: usize,
    pub(super) col_size: usize,
    pub(super) head_size: usize,
    pub(super) inverse_sqrt_head: T,
    pub(super) decode_only_flag: bool,
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
        row_size: usize,
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
            row_size: row_size.max(1),
            col_size: col_size,
            head_size: head_size,
            inverse_sqrt_head: inverse_sqrt_head,
            decode_only_flag: decode_only_flag,
        }
    }

    unsafe fn visit_blocks_for_head(
        &self,
        q_head_ptr: *const T,
        output_head_ptr: *mut T,
        k_head_ptr: *const T,
        v_head_ptr: *const T,
        col_end: usize,
        main_row_range: Option<(usize, usize)>,
        tail_row_range: Option<(usize, usize)>,
    ) {
        let row_step = self.row_size;
        let col_step = self.col_size.max(1);

        if let Some((row_begin, row_end)) = main_row_range {
            for row_chunk in (row_begin..row_end).step_by(row_step) {
                let row_chunk_end = row_chunk + row_step;

                for col_chunk in (0..col_end).step_by(col_step) {
                    let col_chunk_end = (col_chunk + col_step).min(col_end);

                    let _ = (
                        q_head_ptr,
                        output_head_ptr,
                        row_chunk,
                        row_chunk_end,
                        col_chunk,
                        col_chunk_end,
                        k_head_ptr,
                        v_head_ptr,
                    );
                }
            }
        }

        if let Some((tail_begin, tail_end)) = tail_row_range {
            for col_chunk in (0..col_end).step_by(col_step) {
                let col_chunk_end = (col_chunk + col_step).min(col_end);

                let _ = (
                    q_head_ptr,
                    output_head_ptr,
                    tail_begin,
                    tail_end,
                    col_chunk,
                    col_chunk_end,
                    k_head_ptr,
                    v_head_ptr,
                );
            }
        }
    }

    unsafe fn run_sequence_split(
        &self,
        q_slice_ptr: *const T,
        output_slice_ptr: *mut T,
        k_batch_ptr: *const T,
        v_batch_ptr: *const T,
        col_end: usize,
        slice_len: usize,
        aligned_len: usize,
        thread_num: usize,
        thread_id: usize,
        q_head_stride: usize,
        kv_head_stride: usize,
    ) {
        let main_row_range =
            split_sequence_by_triangle(aligned_len, self.row_size, thread_num, thread_id);
        let tail_row_range = if aligned_len < slice_len {
            if thread_num != 0 && thread_id + 1 == thread_num {
                Some((aligned_len, slice_len))
            } else {
                None
            }
        } else {
            None
        };

        for kv_head in 0..self.kv_head_num {
            let q_head_offset = kv_head * q_head_stride;
            let kv_head_offset = kv_head * kv_head_stride;
            let q_head_ptr = q_slice_ptr.add(q_head_offset);
            let output_head_ptr = output_slice_ptr.add(q_head_offset);
            let k_head_ptr = k_batch_ptr.add(kv_head_offset);
            let v_head_ptr = v_batch_ptr.add(kv_head_offset);

            self.visit_blocks_for_head(
                q_head_ptr,
                output_head_ptr,
                k_head_ptr,
                v_head_ptr,
                col_end,
                main_row_range,
                tail_row_range,
            );
        }
    }

    unsafe fn run_head_split(
        &self,
        q_slice_ptr: *const T,
        output_slice_ptr: *mut T,
        k_batch_ptr: *const T,
        v_batch_ptr: *const T,
        col_end: usize,
        slice_len: usize,
        aligned_len: usize,
        thread_num: usize,
        thread_id: usize,
        attention_heads_per_kv: usize,
        kv_head_stride: usize,
    ) {
        let Some((head_begin, head_end)) =
            split_attention_heads(self.attention_head_num, thread_num, thread_id)
        else {
            return;
        };

        let main_row_range = (aligned_len != 0).then_some((0, aligned_len));
        let tail_row_range = (aligned_len < slice_len).then_some((aligned_len, slice_len));

        for attention_head in head_begin..head_end {
            let kv_head = attention_head / attention_heads_per_kv;
            let q_head_offset = attention_head * self.head_size;
            let kv_head_offset = kv_head * kv_head_stride;
            let q_head_ptr = q_slice_ptr.add(q_head_offset);
            let output_head_ptr = output_slice_ptr.add(q_head_offset);
            let k_head_ptr = k_batch_ptr.add(kv_head_offset);
            let v_head_ptr = v_batch_ptr.add(kv_head_offset);

            self.visit_blocks_for_head(
                q_head_ptr,
                output_head_ptr,
                k_head_ptr,
                v_head_ptr,
                col_end,
                main_row_range,
                tail_row_range,
            );
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
            let k_ptr = self.k_ptr.ptr;
            let v_ptr = self.v_ptr.ptr;
            let output_ptr = self.output_ptr.ptr;
            let q_ptr = self.q_ptr.ptr;
            let row_step = self.row_size;
            let kv_batch_stride = self.kv_head_num * self.seq_len * self.head_size;
            let q_token_stride = self.attention_head_num * self.head_size;
            let attention_heads_per_kv = self.attention_head_num / self.kv_head_num;
            let q_head_stride = attention_heads_per_kv * self.head_size;
            let kv_head_stride = self.seq_len * self.head_size;

            for slice in attention_list {
                if slice.batch_index >= self.batch_size {
                    continue;
                }

                let q_slice_ptr = q_ptr.add(slice.token_start_index * q_token_stride);
                let output_slice_ptr = output_ptr.add(slice.token_start_index * q_token_stride);
                let k_batch_ptr = k_ptr.add(slice.batch_index * kv_batch_stride);
                let v_batch_ptr = v_ptr.add(slice.batch_index * kv_batch_stride);
                let col_end = slice.sequence_index + slice.length;
                let aligned_len = slice.length / row_step * row_step;
                let use_head_split =
                    should_split_by_attention_head(slice.length, row_step, thread_num);

                if use_head_split {
                    self.run_head_split(
                        q_slice_ptr,
                        output_slice_ptr,
                        k_batch_ptr,
                        v_batch_ptr,
                        col_end,
                        slice.length,
                        aligned_len,
                        thread_num,
                        thread_id,
                        attention_heads_per_kv,
                        kv_head_stride,
                    );
                } else {
                    self.run_sequence_split(
                        q_slice_ptr,
                        output_slice_ptr,
                        k_batch_ptr,
                        v_batch_ptr,
                        col_end,
                        slice.length,
                        aligned_len,
                        thread_num,
                        thread_id,
                        q_head_stride,
                        kv_head_stride,
                    );
                }
            }
        }
    }
}
