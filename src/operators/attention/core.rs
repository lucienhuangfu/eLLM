use std::ops::{Add, Div, Mul, Sub};

use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};

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
    pub(super) row_step: usize,
    pub(super) col_step: usize,
    pub(super) head_size: usize,
    pub(super) inverse_sqrt_head: T,
    pub(super) decode_only_flag: bool,
    pub(super) thread_num: usize,
    scratch: AttentionScratch<T>,
}

#[derive(Clone)]
pub(super) struct AttentionScratch<T> {
    running_max_pool: Box<[T]>,
    running_max_stride: usize,
    running_denom_pool: Box<[T]>,
    running_denom_stride: usize,
    scores_pool: Box<[T]>,
    scores_stride: usize,
}

pub(super) struct AttentionScratchSlice<'a, T> {
    pub(super) running_max: &'a mut [T],
    pub(super) running_denom: &'a mut [T],
    pub(super) scores: &'a mut [T],
}

#[derive(Copy, Clone)]
pub(super) struct RowVisitPlan {
    pub(super) main: Option<(usize, usize)>,
    pub(super) tail: Option<(usize, usize)>,
}

impl<T> AttentionScratch<T>
where
    T: Copy + Default,
{
    fn new(thread_num: usize, row_step: usize, col_step: usize) -> Self {
        let running_max_stride = row_step.max(1);
        let running_denom_stride = row_step.max(1);
        let scores_stride = col_step.max(1);

        Self {
            running_max_pool: vec![T::default(); thread_num * running_max_stride]
                .into_boxed_slice(),
            running_max_stride,
            running_denom_pool: vec![T::default(); thread_num * running_denom_stride]
                .into_boxed_slice(),
            running_denom_stride,
            scores_pool: vec![T::default(); thread_num * scores_stride].into_boxed_slice(),
            scores_stride,
        }
    }

    #[inline(always)]
    fn thread_buffers(
        &self,
        thread_id: usize,
        row_count: usize,
        col_count: usize,
    ) -> AttentionScratchSlice<'_, T> {
        unsafe {
            let running_max = std::slice::from_raw_parts_mut(
                self.running_max_pool
                    .as_ptr()
                    .add(thread_id * self.running_max_stride) as *mut T,
                row_count,
            );
            let running_denom = std::slice::from_raw_parts_mut(
                self.running_denom_pool
                    .as_ptr()
                    .add(thread_id * self.running_denom_stride) as *mut T,
                row_count,
            );
            let scores = std::slice::from_raw_parts_mut(
                self.scores_pool
                    .as_ptr()
                    .add(thread_id * self.scores_stride) as *mut T,
                col_count,
            );
            AttentionScratchSlice {
                running_max,
                running_denom,
                scores,
            }
        }
    }
}

impl<T> AttentionScratchSlice<'_, T>
where
    T: Copy + Default + NegInfinity,
{
    #[inline(always)]
    pub(super) fn clear(&mut self) {
        for value in self.running_max.iter_mut() {
            *value = T::neg_infinity();
        }
        for value in self.running_denom.iter_mut() {
            *value = T::default();
        }
        for value in self.scores.iter_mut() {
            *value = T::default();
        }
    }
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
