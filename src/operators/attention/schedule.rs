use std::ops::{Add, Div, Mul, Sub};

use super::core::RowVisitPlan;
use super::split_sequence::split_sequence_by_triangle;
use super::Attention;
use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};
use crate::common::sequence_slice::SequenceSlice;

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
            main: split_sequence_by_triangle(aligned_len, self.row_step, thread_num, thread_id),
            tail: if aligned_len < slice_len && thread_num != 0 && thread_id + 1 == thread_num {
                Some((aligned_len, slice_len))
            } else {
                None
            },
        };

        for kv_head in 0..self.kv_head_num {
            let kv_head_offset = kv_head * kv_head_stride;
            let k_head_ptr = k_batch_ptr.add(kv_head_offset);
            let v_head_ptr = v_batch_ptr.add(kv_head_offset);

            for local_head in 0..attention_heads_per_kv {
                let attention_head = kv_head * attention_heads_per_kv + local_head;
                let q_head_offset = attention_head * self.head_size;
                let q_head_ptr = q_slice_ptr.add(q_head_offset);
                let output_head_ptr = output_slice_ptr.add(q_head_offset);

                self.visit_blocks_for_head(
                    q_head_ptr,
                    output_head_ptr,
                    k_head_ptr,
                    v_head_ptr,
                    thread_id,
                    sequence_index,
                    col_end,
                    row_plan,
                );
            }
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

        let active_thread_num = thread_num.min(self.attention_head_num);
        if thread_id >= active_thread_num || active_thread_num == 0 || self.kv_head_num == 0 {
            return;
        }

        let row_plan = RowVisitPlan {
            main: (aligned_len != 0).then_some((0, aligned_len)),
            tail: (aligned_len < slice_len).then_some((aligned_len, slice_len)),
        };

        let kv_heads_per_wave = self.kv_heads_per_wave(active_thread_num, attention_heads_per_kv);
        if kv_heads_per_wave == 0 {
            return;
        }

        for kv_head_begin in (0..self.kv_head_num).step_by(kv_heads_per_wave) {
            let kv_head_end = (kv_head_begin + kv_heads_per_wave).min(self.kv_head_num);
            let slot_num = (kv_head_end - kv_head_begin) * attention_heads_per_kv;
            let active_threads_this_wave = active_thread_num.min(slot_num);

            if thread_id >= active_threads_this_wave {
                continue;
            }

            let Some((slot_begin, slot_end)) =
                Self::split_contiguous_range(slot_num, active_threads_this_wave, thread_id)
            else {
                continue;
            };

            for slot in slot_begin..slot_end {
                let kv_head = kv_head_begin + slot / attention_heads_per_kv;
                let local_head = slot % attention_heads_per_kv;
                let kv_head_offset = kv_head * kv_head_stride;
                let k_head_ptr = k_batch_ptr.add(kv_head_offset);
                let v_head_ptr = v_batch_ptr.add(kv_head_offset);
                let attention_head = kv_head * attention_heads_per_kv + local_head;
                let q_head_offset = attention_head * self.head_size;
                let q_head_ptr = q_slice_ptr.add(q_head_offset);
                let output_head_ptr = output_slice_ptr.add(q_head_offset);

                self.visit_blocks_for_head(
                    q_head_ptr,
                    output_head_ptr,
                    k_head_ptr,
                    v_head_ptr,
                    thread_id,
                    sequence_index,
                    col_end,
                    row_plan,
                );
            }
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
                let aligned_len = slice.length / self.row_step * self.row_step;
                let use_head_split = slice.length > 0
                    && thread_num > 0
                    && slice.length.div_ceil(self.row_step.max(1)) < thread_num;

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

#[cfg(test)]
mod tests {
    use super::Attention;
    use crate::common::sequence_slice::SequenceSlice;

    fn naive_attention_row(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        head_size: usize,
        visible_col_end: usize,
        inverse_sqrt_head: f32,
    ) -> Vec<f32> {
        let mut max_score = f32::NEG_INFINITY;
        let mut scores = vec![0.0; visible_col_end];

        for col in 0..visible_col_end {
            let mut score = 0.0;
            for index in 0..head_size {
                score += q[index] * k[col * head_size + index];
            }
            score *= inverse_sqrt_head;
            scores[col] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let mut denom = 0.0;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            denom += *score;
        }

        let mut output = vec![0.0; head_size];
        for col in 0..visible_col_end {
            let weight = scores[col] / denom;
            for index in 0..head_size {
                output[index] += weight * v[col * head_size + index];
            }
        }
        output
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
            assert!((actual_value - expected_value).abs() < 1e-5);
        }
    }

    #[test]
    fn sequence_split_leaves_tail_rows_empty() {
        let head_size = 2;
        let inverse_sqrt_head = 1.0;
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![-1.0; q.len()];

        let attention = Attention::new(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            output.as_mut_ptr(),
            1,
            1,
            1,
            3,
            2,
            2,
            head_size,
            inverse_sqrt_head,
            false,
            1,
        );

        let slices = [SequenceSlice {
            token_start_index: 0,
            batch_index: 0,
            sequence_index: 0,
            length: 3,
        }];

        attention.run(0, 0, &slices, 1, 0);

        for row in 0..2 {
            let expected = naive_attention_row(
                &q[row * head_size..(row + 1) * head_size],
                &k,
                &v,
                head_size,
                row + 1,
                inverse_sqrt_head,
            );
            let actual = &output[row * head_size..(row + 1) * head_size];
            assert_close(actual, &expected);
        }

        assert_close(&output[2 * head_size..3 * head_size], &[0.0, 0.0]);
    }
}
