use std::ops::{Add, Div, Mul, Sub};

use super::run::RowVisitPlan;
use super::Attention;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};

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
        col_end: usize,
        row_plan: RowVisitPlan,
    ) {
        let row_step = self.row_size;
        let col_step = self.col_size.max(1);

        if let Some((row_begin, row_end)) = row_plan.main {
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

        if let Some((tail_begin, tail_end)) = row_plan.tail {
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
                col_end,
                row_plan,
            );
        }
    }
}
