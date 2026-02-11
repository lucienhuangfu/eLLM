use crate::common::num_traits::Exp;
use std::ops::{AddAssign, Div, Sub};
use std::ptr;

pub fn experts_topk_softmax_norm<
    T: Exp + Default + AddAssign + PartialOrd + Copy + Sub<Output = T> + Div<Output = T>,
>(
    input_ptr: *const T,
    topk_values_ptr: *mut T,
    topk_indices_ptr: *mut usize,
    // [num_experts]
    experts_indicator_ptr: *mut bool,
    // [num_experts, prefill_size]
    indices_ptr: *mut bool,
    value_ptr: *mut T,
    index_token: usize,
    num_token: usize,
    num_experts: usize,
    num_topk: usize,
    norm_topk_prob: bool,
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
        let topk_items = &indexed_values[..(num_topk).min(num_experts)];

        // Calculate sum of all input values for softmax normalization
        let mut sum = T::default();
        for &value in input_slice {
            sum += value.exp();
        }

        // Set experts_indicator for top-k experts
        for &(expert_idx, _) in topk_items {
            *experts_indicator_ptr.add(expert_idx) = true;
        }

        let mut norm_sum = T::default();
        for (k, &(expert_idx, value)) in topk_items.iter().enumerate() {
            let softmax_value = value.exp() / sum;
            // if norm_topk_prob {
            //     norm_sum += softmax_value;
            // }
            // Store top-k values and indices
            *topk_values_ptr.add(k) = softmax_value;
            *topk_indices_ptr.add(k) = expert_idx;
        }

        if norm_topk_prob {
            let mut norm_sum = T::default();
            for i in 0..num_topk {
                norm_sum += *topk_values_ptr.add(i);
            }
            // let scale = 1.0 / norm_sum;
            for i in 0..num_topk {
                let val = *topk_values_ptr.add(i);
                *topk_values_ptr.add(i) = val / norm_sum;
            }
        }

        // For each top-k expert, compute softmax and set outputs
        for (k, &(expert_idx, _)) in topk_items.iter().enumerate() {
            // Compute softmax: exp(x) / sum
            let mut softmax_value = *topk_values_ptr.add(k);
            // if norm_topk_prob {
            //     softmax_value = softmax_value / norm_sum;
            //}
            *experts_indicator_ptr.add(expert_idx) = true;
            // Set indices_ptr at [expert_idx * num_token + index_token]
            let offset = (expert_idx) * (num_token) + (index_token);
            *indices_ptr.add(offset) = true;
            // Set value_ptr at the same position as indices
            *value_ptr.add(offset) = softmax_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_ulps_eq};

    #[test]
    fn test_experts_topk_softmax_norm_basic() {
        const NUM_EXPERTS: usize = 16;
        const NUM_TOPK: usize = 4;
        const NUM_TOKEN: usize = 3;
        const INDEX_TOKEN: usize = 1;
        let data = [
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        ];
        let mut experts_indicator = [false; NUM_EXPERTS];
        let mut topk_vals = [0.0f32; NUM_TOPK];
        let mut topk_idx = [0; NUM_TOPK];
        let mut indices = [false; (NUM_EXPERTS * NUM_TOKEN)];
        let mut values = [0.0f32; (NUM_EXPERTS * NUM_TOKEN)];

        unsafe {
            super::experts_topk_softmax_norm(
                data.as_ptr(),
                topk_vals.as_mut_ptr(),
                topk_idx.as_mut_ptr(),
                experts_indicator.as_mut_ptr(),
                indices.as_mut_ptr(),
                values.as_mut_ptr(),
                INDEX_TOKEN,
                NUM_TOKEN,
                NUM_EXPERTS,
                NUM_TOPK,
                true,
            );
        }

        let mut expected: Vec<(usize, f32)> = data
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, val)| (idx, val))
            .collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1));

        let denom: f32 = expected.iter().take(NUM_TOPK).map(|(_, v)| v.exp()).sum();
        let mut is_topk = [false; NUM_EXPERTS];

        for i in 0..(NUM_TOPK) {
            let (idx, val) = expected[i];
            // Store top-k values and indices
            let offset = (idx) * (NUM_TOKEN) + (INDEX_TOKEN);
            assert!(experts_indicator[idx]);
            assert!(indices[offset]);
            assert_relative_eq!(values[offset], val.exp() / denom, epsilon = 1e-6);
            is_topk[idx] = true;
        }

        for expert in 0..(NUM_EXPERTS) {
            if !is_topk[expert] {
                assert!(!experts_indicator[expert]);
            }
            for token in 0..(NUM_TOKEN) {
                let offset = expert * (NUM_TOKEN) + token;
                if is_topk[expert] && token == (INDEX_TOKEN) {
                    continue;
                }
                assert!(!indices[offset]);
                assert_relative_eq!(values[offset], 0.0, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_experts_topk_softmax_norm_topk_exceeds_num_experts() {
        const NUM_EXPERTS: usize = 3;
        const NUM_TOPK: usize = 5;
        const NUM_TOKEN: usize = 2;
        const INDEX_TOKEN: usize = 0;
        let data = [1.0f32, -0.75, 2.5];
        let mut experts_indicator = [false; NUM_EXPERTS];
        let mut topk_vals = [0.0f32; NUM_TOPK];
        let mut topk_idx = [0; NUM_TOPK];
        let mut indices = [false; (NUM_EXPERTS * NUM_TOKEN)];
        let mut values = [0.0f32; (NUM_EXPERTS * NUM_TOKEN)];

        unsafe {
            super::experts_topk_softmax_norm(
                data.as_ptr(),
                topk_vals.as_mut_ptr(),
                topk_idx.as_mut_ptr(),
                experts_indicator.as_mut_ptr(),
                indices.as_mut_ptr(),
                values.as_mut_ptr(),
                INDEX_TOKEN,
                NUM_TOKEN,
                NUM_EXPERTS,
                NUM_TOPK,
                true,
            );
        }

        let mut expected: Vec<(usize, f32)> = data
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, val)| (idx, val))
            .collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1));

        let denom: f32 = data.iter().map(|v| v.exp()).sum();

        for (idx, &val) in data.iter().enumerate() {
            let offset = idx * (NUM_TOKEN) + (INDEX_TOKEN);
            assert!(experts_indicator[idx]);
            assert!(indices[offset]);
            assert_relative_eq!(values[offset], val.exp() / denom, epsilon = 1e-6);
        }

        for token in 0..(NUM_TOKEN) {
            if token == (INDEX_TOKEN) {
                continue;
            }
            for expert in 0..(NUM_EXPERTS) {
                let offset = expert * (NUM_TOKEN) + token;
                assert!(!indices[offset]);
                assert_relative_eq!(values[offset], 0.0, epsilon = 1e-6);
            }
        }
    }
}






