use std::ops::{Add, Div, Mul, Sub};

use crate::common::num_traits::NegInfinity;
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};

use super::scratch::{AttentionScratch, AttentionScratchSlice};

#[derive(Clone)]
pub struct Attention<T> {
    pub(super) q_ptr: ConstPtr<T>,
    pub(super) k_ptr: ConstPtr<T>,
    pub(super) v_ptr: ConstPtr<T>,
    pub(super) output_ptr: MutPtr<T>,
    pub(super) seq_len: usize,
    pub(super) batch_size: usize,
    pub(super) attention_head_num: usize,
    pub(super) kv_head_num: usize,
    pub(super) k_batch_stride: usize,
    pub(super) k_head_stride: usize,
    pub(super) k_seq_stride: usize,
    pub(super) v_batch_stride: usize,
    pub(super) v_head_stride: usize,
    pub(super) v_seq_stride: usize,
    pub(super) head_size: usize,
    pub(super) inverse_sqrt_head: T,
    pub(super) row_step: usize,
    pub(super) col_step: usize,
    pub(super) decode_only_flag: bool,
    pub(super) thread_num: usize,
    scratch: AttentionScratch<T>,
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
        + NegInfinity,
{
    pub fn new(
        q_ptr: *const T,
        k_ptr: *const T,
        v_ptr: *const T,
        output_ptr: *mut T,
        seq_len: usize,
        batch_size: usize,
        attention_head_num: usize,
        kv_head_num: usize,
        k_batch_stride: usize,
        k_head_stride: usize,
        k_seq_stride: usize,
        v_batch_stride: usize,
        v_head_stride: usize,
        v_seq_stride: usize,
        row_step: usize,
        col_step: usize,
        head_size: usize,
        inverse_sqrt_head: T,
        decode_only_flag: bool,
        thread_num: usize,
    ) -> Self {
        let thread_num = thread_num.max(1);
        let row_step = row_step.max(1);

        Self {
            q_ptr: ConstPtr { ptr: q_ptr },
            k_ptr: ConstPtr { ptr: k_ptr },
            v_ptr: ConstPtr { ptr: v_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            batch_size,
            attention_head_num,
            kv_head_num,
            k_batch_stride,
            k_head_stride,
            k_seq_stride,
            v_batch_stride,
            v_head_stride,
            v_seq_stride,
            seq_len,
            row_step,
            col_step,
            head_size,
            inverse_sqrt_head,
            decode_only_flag,
            thread_num,
            scratch: AttentionScratch::new(thread_num, row_step, col_step),
        }
    }

    #[inline(always)]
    pub(super) fn thread_buffers(
        &self,
        thread_id: usize,
        row_count: usize,
        col_count: usize,
    ) -> AttentionScratchSlice<'_, T> {
        debug_assert!(thread_id < self.thread_num);
        self.scratch.thread_buffers(thread_id, row_count, col_count)
    }

    #[inline]
    pub(super) fn split_contiguous_range(
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
    pub(super) fn kv_heads_per_wave(
        &self,
        active_thread_num: usize,
        attention_heads_per_kv: usize,
    ) -> usize {
        if active_thread_num == 0 || attention_heads_per_kv == 0 || self.kv_head_num == 0 {
            return 0;
        }

        active_thread_num
            .div_ceil(attention_heads_per_kv)
            .max(1)
            .min(self.kv_head_num)
    }
}
