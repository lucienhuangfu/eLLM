use std::ops::{Add, Div, Mul, Sub};

use crate::common::num_traits::{exp::Exp, neg_infinity::NegInfinity};

pub fn block_flash_attention<T>(
    q_head_ptr: *const T,
    output_head_ptr: *mut T,
    row_begin: usize,
    row_end: usize,
    col_begin: usize,
    col_end: usize,
    total_col_end: usize,
    k_head_ptr: *const T,
    v_head_ptr: *const T,
    head_size: usize,
    inverse_sqrt_head: T,
    sequence_index: usize,
    running_max: &mut [T],
    running_denom: &mut [T],
    scores: &mut [T],
) where
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
    unsafe {
        for (row_offset, row) in (row_begin..row_end).enumerate() {
            let visible_col_end = (sequence_index + row + 1).min(total_col_end);
            let row_col_end = col_end.min(visible_col_end);
            if col_begin >= row_col_end {
                continue;
            }

            let head_offset = row * head_size;
            let q_row_ptr = q_head_ptr.add(head_offset);
            let output_row_ptr = output_head_ptr.add(head_offset);
            let block_len = row_col_end - col_begin;
            let mut block_max = T::neg_infinity();

            for offset in 0..block_len {
                let col = col_begin + offset;
                let key_row_ptr = k_head_ptr.add(col * head_size);

                let mut score = T::default();
                for index in 0..head_size {
                    score = score + *q_row_ptr.add(index) * *key_row_ptr.add(index);
                }
                score = score * inverse_sqrt_head;
                scores[offset] = score;
                if score > block_max {
                    block_max = score;
                }
            }

            let next_max = if block_max > running_max[row_offset] {
                block_max
            } else {
                running_max[row_offset]
            };

            let carry = running_denom[row_offset] * (running_max[row_offset] - next_max).exp();
            let mut next_denom = carry;
            for offset in 0..block_len {
                next_denom = next_denom + (scores[offset] - next_max).exp();
            }

            let previous_weight = carry / next_denom;
            for index in 0..head_size {
                *output_row_ptr.add(index) = *output_row_ptr.add(index) * previous_weight;
            }

            for offset in 0..block_len {
                let col = col_begin + offset;
                let value_row_ptr = v_head_ptr.add(col * head_size);
                let weight = (scores[offset] - next_max).exp() / next_denom;
                for index in 0..head_size {
                    *output_row_ptr.add(index) =
                        *output_row_ptr.add(index) + *value_row_ptr.add(index) * weight;
                }
            }

            running_max[row_offset] = next_max;
            running_denom[row_offset] = next_denom;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::block_flash_attention;

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
        for (index, (actual_value, expected_value)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                (actual_value - expected_value).abs() < 1e-5,
                "mismatch at index {index}: actual={actual_value}, expected={expected_value}"
            );
        }
    }

    #[test]
    fn test_block_flash_attention_matches_naive_attention() {
        let head_size = 3;
        let row_begin = 0;
        let row_end = 3;
        let col_end = 4;
        let col_size = 2;
        let inverse_sqrt_head = 0.5;

        let q = vec![1.0, 0.0, 1.0, 0.5, 1.0, -0.5, 1.5, -0.5, 0.25];
        let k = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 1.0, 1.0, 0.0, -0.5, 0.25, 1.5];
        let v = vec![
            1.0, 2.0, 3.0, 0.5, 1.5, -1.0, -2.0, 0.0, 1.0, 4.0, -1.0, 0.5,
        ];
        let mut blocked_output = vec![0.0; q.len()];
        let row_count = row_end - row_begin;
        let mut running_max = vec![f32::NEG_INFINITY; row_count];
        let mut running_denom = vec![0.0; row_count];
        let mut scores = vec![0.0; col_size];

        for col_begin in (0..col_end).step_by(col_size) {
            let block_end = (col_begin + col_size).min(col_end);
            block_flash_attention(
                q.as_ptr(),
                blocked_output.as_mut_ptr(),
                row_begin,
                row_end,
                col_begin,
                block_end,
                col_end,
                k.as_ptr(),
                v.as_ptr(),
                head_size,
                inverse_sqrt_head,
                0,
                &mut running_max,
                &mut running_denom,
                &mut scores,
            );
        }

        for row in row_begin..row_end {
            let expected = naive_attention_row(
                &q[row * head_size..(row + 1) * head_size],
                &k,
                &v,
                head_size,
                row + 1,
                inverse_sqrt_head,
            );
            let actual = &blocked_output[row * head_size..(row + 1) * head_size];
            assert_close(actual, &expected);
        }
    }

    #[test]
    fn test_block_flash_attention_respects_row_range_and_sequence_offset() {
        let head_size = 2;
        let row_begin = 1;
        let row_end = 3;
        let col_end = 6;
        let col_size = 3;
        let inverse_sqrt_head = 1.0;
        let sequence_index = 2;

        let q = vec![9.0, 9.0, 1.0, 0.0, 0.0, 1.0, 8.0, 8.0];
        let k = vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5, 0.5, -1.0, 2.0, 0.25,
        ];
        let v = vec![0.0, 1.0, 1.0, 0.0, 2.0, 2.0, -1.0, 3.0, 4.0, -2.0, 0.5, 0.5];
        let mut output = vec![-7.0; q.len()];

        for row in row_begin..row_end {
            let row_offset = row * head_size;
            for index in 0..head_size {
                output[row_offset + index] = 0.0;
            }
        }

        let row_count = row_end - row_begin;
        let mut running_max = vec![f32::NEG_INFINITY; row_count];
        let mut running_denom = vec![0.0; row_count];
        let mut scores = vec![0.0; col_size];

        for col_begin in (0..col_end).step_by(col_size) {
            let block_end = (col_begin + col_size).min(col_end);
            block_flash_attention(
                q.as_ptr(),
                output.as_mut_ptr(),
                row_begin,
                row_end,
                col_begin,
                block_end,
                col_end,
                k.as_ptr(),
                v.as_ptr(),
                head_size,
                inverse_sqrt_head,
                sequence_index,
                &mut running_max,
                &mut running_denom,
                &mut scores,
            );
        }

        assert_close(&output[0..head_size], &[-7.0, -7.0]);
        assert_close(&output[3 * head_size..4 * head_size], &[-7.0, -7.0]);

        for row in row_begin..row_end {
            let visible_col_end = (sequence_index + row + 1).min(col_end);
            let expected = naive_attention_row(
                &q[row * head_size..(row + 1) * head_size],
                &k,
                &v,
                head_size,
                visible_col_end,
                inverse_sqrt_head,
            );
            let actual = &output[row * head_size..(row + 1) * head_size];
            assert_close(actual, &expected);
        }
    }
}
