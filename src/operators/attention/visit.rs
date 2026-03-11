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
    unsafe fn block_flash_attention(
        &self,
        q_head_ptr: *const T,
        output_head_ptr: *mut T,
        row_begin: usize,
        row_end: usize,
        col_end: usize,
        k_head_ptr: *const T,
        v_head_ptr: *const T,
        sequence_index: usize,
    ) {
        let visible_row_end = row_end.min(col_end.saturating_sub(sequence_index));
        for row in row_begin..visible_row_end {
            let row_offset = row * self.head_size;
            self.compute(
                q_head_ptr.add(row_offset),
                k_head_ptr,
                v_head_ptr,
                output_head_ptr.add(row_offset),
                sequence_index + row,
            );
        }
    }

    pub(super) unsafe fn visit_blocks_for_head(
        &self,
        q_head_ptr: *const T,
        output_head_ptr: *mut T,
        k_head_ptr: *const T,
        v_head_ptr: *const T,
        sequence_index: usize,
        col_end: usize,
        row_plan: RowVisitPlan,
    ) {
        let row_step = self.row_size;

        if let Some((row_begin, row_end)) = row_plan.main {
            for row_chunk in (row_begin..row_end).step_by(row_step) {
                let row_chunk_end = row_chunk + row_step;

                self.block_flash_attention(
                    q_head_ptr,
                    output_head_ptr,
                    row_chunk,
                    row_chunk_end,
                    col_end,
                    k_head_ptr,
                    v_head_ptr,
                    sequence_index,
                );
            }
        }

        if let Some((tail_begin, tail_end)) = row_plan.tail {
            self.block_flash_attention(
                q_head_ptr,
                output_head_ptr,
                tail_begin,
                tail_end,
                col_end,
                k_head_ptr,
                v_head_ptr,
                sequence_index,
            );
        }
    }

    pub(super) unsafe fn visit_heads_for_kv(
        &self,
        q_slice_ptr: *const T,
        output_slice_ptr: *mut T,
        k_batch_ptr: *const T,
        v_batch_ptr: *const T,
        kv_head: usize,
        local_head_range: std::ops::Range<usize>,
        kv_head_stride: usize,
        attention_heads_per_kv: usize,
        sequence_index: usize,
        col_end: usize,
        row_plan: RowVisitPlan,
    ) {
        let kv_head_offset = kv_head * kv_head_stride;
        let k_head_ptr = k_batch_ptr.add(kv_head_offset);
        let v_head_ptr = v_batch_ptr.add(kv_head_offset);

        for local_head in local_head_range {
            let attention_head = kv_head * attention_heads_per_kv + local_head;
            let q_head_offset = attention_head * self.head_size;
            let q_head_ptr = q_slice_ptr.add(q_head_offset);
            let output_head_ptr = output_slice_ptr.add(q_head_offset);

            self.visit_blocks_for_head(
                q_head_ptr,
                output_head_ptr,
                k_head_ptr,
                v_head_ptr,
                sequence_index,
                col_end,
                row_plan,
            );
        }
    }
}
