use std::ops::{Add, Div, Mul, Sub};

use super::run::RowVisitPlan;
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
        let row_step = self.row_size;
        let col_step = self.col_size.max(1);

        if let Some((row_begin, row_end)) = row_plan.main {
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
    }
}
