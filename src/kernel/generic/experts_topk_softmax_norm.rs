use crate::kernel::generic::exp::Exp;
use std::ops::{AddAssign, Div, Sub};
use std::ptr;

pub fn experts_topk_softmax_norm<
    T: Exp + Default + AddAssign + PartialOrd + Copy + Sub<Output = T> + Div<Output = T>,
>(
    input_ptr: *const T,
    // [num_experts]
    experts_indicator_ptr: *mut bool,
    // token_size = sequence_chunk_size * batch_size
    // [num_experts, token_size]
    indices_ptr: *mut bool,
    value_ptr: *mut T,
    index_token: usize,
    num_token: usize,
    num_experts: usize,
    num_topk: usize,
) {
    unsafe {
        // Read input values
        let input_slice = std::slice::from_raw_parts(input_ptr, num_experts);

        // Find top-k indices and values
        let mut indexed_values: Vec<(usize, T)> = input_slice
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Sort by value in descending order
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-k
        let topk_items = &indexed_values[..num_topk.min(num_experts)];

        // Calculate sum of all input values for softmax normalization
        let mut sum = T::default();
        for &value in input_slice {
            sum += value.exp();
        }

        // Set experts_indicator for top-k experts
        for &(expert_idx, _) in topk_items {
            *experts_indicator_ptr.add(expert_idx) = true;
        }

        // For each top-k expert, compute softmax and set outputs
        for (k, &(expert_idx, value)) in topk_items.iter().enumerate() {
            // Compute softmax: exp(x) / sum
            let softmax_value = value.exp() / sum;

            *experts_indicator_ptr.add(expert_idx) = true;
            // Set indices_ptr at [expert_idx * num_token + index_token]
            *indices_ptr.add(expert_idx * num_token + index_token) = true;
            // Set value_ptr at the same position as indices
            *value_ptr.add(expert_idx * num_token + index_token) = softmax_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_experts_topk_softmax_norm() {
        // Test data: 128 experts with random-like values
        let mut input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1) % 10.0).collect();
        // Make some values clearly larger for predictable top-k
        input[120] = 15.0; // largest
        input[100] = 14.0;
        input[80] = 13.0;
        input[60] = 12.0;
        input[40] = 11.0;
        input[20] = 10.5;
        input[10] = 10.2;
        input[5] = 10.1; // 8th largest

        let num_experts = 128;
        let num_topk = 8;
        let num_token = 32;
        let index_token = 5;

        let mut experts_indicator = vec![false; num_experts];
        let mut indices = vec![false; num_experts * num_token];
        let mut values = vec![0.0f32; num_experts * num_token];

        experts_topk_softmax_norm(
            input.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indices.as_mut_ptr(),
            values.as_mut_ptr(),
            index_token,
            num_token,
            num_experts,
            num_topk,
        );

        // Expected top-8 experts: 120, 100, 80, 60, 40, 20, 10, 5
        let expected_experts = [120, 100, 80, 60, 40, 20, 10, 5];
        for &expert_idx in &expected_experts {
            assert!(
                experts_indicator[expert_idx],
                "Expert {} should be selected",
                expert_idx
            );
        }

        // Check that only top-k experts are marked
        let true_count = experts_indicator.iter().filter(|&&x| x).count();
        assert_eq!(true_count, num_topk);

        // Check indices array for expected experts
        for &expert_idx in &expected_experts {
            assert!(
                indices[expert_idx * num_token + index_token],
                "Index for expert {} at token {} should be true",
                expert_idx,
                index_token
            );
        }

        // Check that softmax values are written at correct positions
        for &expert_idx in &expected_experts {
            let value_idx = expert_idx * num_token + index_token;
            assert!(
                values[value_idx] > 0.0,
                "Value for expert {} should be non-zero",
                expert_idx
            );
        }

        // Check that softmax values sum to less than 1 (since we use full sum for normalization)
        let mut sum_values = 0.0f32;
        for &expert_idx in &expected_experts {
            sum_values += values[expert_idx * num_token + index_token];
        }
        assert!(sum_values > 0.0 && sum_values < 1.0);

        // Check that values are in descending order by checking actual positions
        for i in 0..num_topk - 1 {
            let val1 = values[expected_experts[i] * num_token + index_token];
            let val2 = values[expected_experts[i + 1] * num_token + index_token];
            assert!(val1 >= val2, "Values should be in descending order");
        }

        // Check that non-selected experts have zero values
        for expert_idx in 0..num_experts {
            if !expected_experts.contains(&expert_idx) {
                let value_idx = expert_idx * num_token + index_token;
                assert_eq!(
                    values[value_idx], 0.0,
                    "Non-selected expert {} should have zero value",
                    expert_idx
                );
            }
        }
    }
}
