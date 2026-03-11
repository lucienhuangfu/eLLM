use std::ops::{Add, Div, Mul, Sub};

use super::split_sequence::split_sequence_by_triangle;
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

#[derive(Copy, Clone)]
pub(super) struct RowVisitPlan {
    pub(super) main: Option<(usize, usize)>,
    pub(super) tail: Option<(usize, usize)>,
}

#[derive(Copy, Clone)]
struct KvHeadThreadAssignment {
    kv_head: usize,
    local_thread_id: usize,
    threads_for_kv: usize,
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
            batch_size,
            attention_head_num,
            kv_head_num,
            seq_len,
            row_size: row_size.max(1),
            col_size,
            head_size,
            inverse_sqrt_head,
            decode_only_flag,
        }
    }

    #[inline]
    fn split_contiguous_range(
        total: usize,
        part_num: usize,
        part_id: usize,
    ) -> Option<(usize, usize)> {
        if total == 0 || part_num == 0 || part_id >= part_num {
            return None;
        }

        let begin = total * part_id / part_num;
        let end = total * (part_id + 1) / part_num;
        (begin < end).then_some((begin, end))
    }

    #[inline]
    fn assign_thread_to_kv_head(
        &self,
        active_thread_num: usize,
        thread_id: usize,
    ) -> Option<KvHeadThreadAssignment> {
        if thread_id >= active_thread_num || self.kv_head_num == 0 {
            return None;
        }

        let base_threads_per_kv = active_thread_num / self.kv_head_num;
        let extra_threads = active_thread_num % self.kv_head_num;
        let extra_thread_span = extra_threads * (base_threads_per_kv + 1);

        let (kv_head, local_thread_id, threads_for_kv) = if thread_id < extra_thread_span {
            let threads_for_kv = base_threads_per_kv + 1;
            (
                thread_id / threads_for_kv,
                thread_id % threads_for_kv,
                threads_for_kv,
            )
        } else {
            let compact_thread_id = thread_id - extra_thread_span;
            (
                extra_threads + compact_thread_id / base_threads_per_kv,
                compact_thread_id % base_threads_per_kv,
                base_threads_per_kv,
            )
        };

        Some(KvHeadThreadAssignment {
            kv_head,
            local_thread_id,
            threads_for_kv,
        })
    }

    unsafe fn run_sequence_split(
        &self,
        q_slice_ptr: *const T,
        output_slice_ptr: *mut T,
        k_batch_ptr: *const T,
        v_batch_ptr: *const T,
        sequence_index: usize,
        col_end: usize,
        slice_len: usize,
        aligned_len: usize,
        thread_num: usize,
        thread_id: usize,
        kv_head_stride: usize,
        attention_heads_per_kv: usize,
    ) {
        let row_plan = RowVisitPlan {
            main: split_sequence_by_triangle(aligned_len, self.row_size, thread_num, thread_id),
            tail: if aligned_len < slice_len && thread_num != 0 && thread_id + 1 == thread_num {
                Some((aligned_len, slice_len))
            } else {
                None
            },
        };

        for kv_head in 0..self.kv_head_num {
            self.visit_heads_for_kv(
                q_slice_ptr,
                output_slice_ptr,
                k_batch_ptr,
                v_batch_ptr,
                kv_head,
                0..attention_heads_per_kv,
                kv_head_stride,
                attention_heads_per_kv,
                sequence_index,
                col_end,
                row_plan,
            );
        }
    }

    unsafe fn run_head_split(
        &self,
        q_slice_ptr: *const T,
        output_slice_ptr: *mut T,
        k_batch_ptr: *const T,
        v_batch_ptr: *const T,
        sequence_index: usize,
        col_end: usize,
        slice_len: usize,
        aligned_len: usize,
        thread_num: usize,
        thread_id: usize,
        attention_heads_per_kv: usize,
        kv_head_stride: usize,
    ) {
        if thread_num == 0 || thread_id >= thread_num || attention_heads_per_kv == 0 {
            return;
        }

        let row_plan = RowVisitPlan {
            main: (aligned_len != 0).then_some((0, aligned_len)),
            tail: (aligned_len < slice_len).then_some((aligned_len, slice_len)),
        };

        if self.kv_head_num >= thread_num {
            let Some((kv_head_begin, kv_head_end)) =
                Self::split_contiguous_range(self.kv_head_num, thread_num, thread_id)
            else {
                return;
            };

            for kv_head in kv_head_begin..kv_head_end {
                self.visit_heads_for_kv(
                    q_slice_ptr,
                    output_slice_ptr,
                    k_batch_ptr,
                    v_batch_ptr,
                    kv_head,
                    0..attention_heads_per_kv,
                    kv_head_stride,
                    attention_heads_per_kv,
                    sequence_index,
                    col_end,
                    row_plan,
                );
            }
            return;
        }

        let active_thread_num = thread_num.min(self.attention_head_num);
        let Some(assignment) = self.assign_thread_to_kv_head(active_thread_num, thread_id) else {
            return;
        };

        let Some((local_head_begin, local_head_end)) = Self::split_contiguous_range(
            attention_heads_per_kv,
            assignment.threads_for_kv,
            assignment.local_thread_id,
        ) else {
            return;
        };

        self.visit_heads_for_kv(
            q_slice_ptr,
            output_slice_ptr,
            k_batch_ptr,
            v_batch_ptr,
            assignment.kv_head,
            local_head_begin..local_head_end,
            kv_head_stride,
            attention_heads_per_kv,
            sequence_index,
            col_end,
            row_plan,
        );
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
            let kv_batch_stride = self.kv_head_num * self.seq_len * self.head_size;
            let q_token_stride = self.attention_head_num * self.head_size;
            let attention_heads_per_kv = self.attention_head_num / self.kv_head_num;
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
                let aligned_len = slice.length / self.row_size * self.row_size;
                let use_head_split = slice.length > 0
                    && thread_num > 0
                    && slice.length.div_ceil(self.row_size.max(1)) < thread_num;

                if use_head_split {
                    self.run_head_split(
                        q_slice_ptr,
                        output_slice_ptr,
                        k_batch_ptr,
                        v_batch_ptr,
                        slice.sequence_index,
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
                        slice.sequence_index,
                        col_end,
                        slice.length,
                        aligned_len,
                        thread_num,
                        thread_id,
                        kv_head_stride,
                        attention_heads_per_kv,
                    );
                }
            }
        }
    }
}
