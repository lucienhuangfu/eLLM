use std::ops::{Add, Div, Mul, Sub};

use super::core::RowVisitPlan;
use super::Attention;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::operators::traits::AttentionTrait;

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
    #[inline(always)]
    unsafe fn visit_aligned_row_range(
        &self,
        q_head_ptr: *const T,
        output_head_ptr: *mut T,
        k_head_ptr: *const T,
        v_head_ptr: *const T,
        thread_id: usize,
        sequence_index: usize,
        col_end: usize,
        row_begin: usize,
        row_end: usize,
    ) {
        let row_step = self.row_step;
        let col_step = self.col_step.max(1);

        for row_chunk in (row_begin..row_end).step_by(row_step) {
            let row_chunk_end = row_chunk + row_step;
            let visible_row_end = row_chunk_end.min(col_end.saturating_sub(sequence_index));
            if row_chunk >= visible_row_end {
                continue;
            }

            let row_count = visible_row_end - row_chunk;
            let mut scratch = self.thread_buffers(thread_id, row_count, col_step);
            scratch.clear();

            for row in row_chunk..visible_row_end {
                let row_offset = row * self.head_size;
                for index in 0..self.head_size {
                    *output_head_ptr.add(row_offset + index) = T::default();
                }
            }

            for col_begin in (0..col_end).step_by(col_step) {
                let col_chunk_end = (col_begin + col_step).min(col_end);
                AttentionTrait::compute(
                    self,
                    q_head_ptr,
                    k_head_ptr,
                    v_head_ptr,
                    output_head_ptr,
                    row_chunk,
                    visible_row_end,
                    col_begin,
                    col_chunk_end,
                    col_end,
                    sequence_index,
                    scratch.running_max,
                    scratch.running_denom,
                    scratch.scores,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn visit_tail_row_range(
        &self,
        _q_head_ptr: *const T,
        output_head_ptr: *mut T,
        _k_head_ptr: *const T,
        _v_head_ptr: *const T,
        _thread_id: usize,
        sequence_index: usize,
        col_end: usize,
        row_begin: usize,
        row_end: usize,
    ) {
        let visible_row_end = row_end.min(col_end.saturating_sub(sequence_index));
        if row_begin >= visible_row_end {
            return;
        }

        // The trailing partial block cannot use the current compute kernel yet, so leave it zeroed.
        for row in row_begin..visible_row_end {
            let row_offset = row * self.head_size;
            for index in 0..self.head_size {
                *output_head_ptr.add(row_offset + index) = T::default();
            }
        }
    }

    pub(super) unsafe fn visit_blocks_for_head(
        &self,
        q_head_ptr: *const T,
        output_head_ptr: *mut T,
        k_head_ptr: *const T,
        v_head_ptr: *const T,
        thread_id: usize,
        sequence_index: usize,
        col_end: usize,
        row_plan: RowVisitPlan,
    ) {
        if let Some((row_begin, row_end)) = row_plan.main {
            self.visit_aligned_row_range(
                q_head_ptr,
                output_head_ptr,
                k_head_ptr,
                v_head_ptr,
                thread_id,
                sequence_index,
                col_end,
                row_begin,
                row_end,
            );
        }

        if let Some((row_begin, row_end)) = row_plan.tail {
            self.visit_tail_row_range(
                q_head_ptr,
                output_head_ptr,
                k_head_ptr,
                v_head_ptr,
                thread_id,
                sequence_index,
                col_end,
                row_begin,
                row_end,
            );
        }
    }
}
