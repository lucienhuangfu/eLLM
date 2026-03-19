use crate::common::heap::FixedMinHeap;
use std::ops::{AddAssign, Div};

pub fn experts_topk_norm<T>(
    ptr1: *const T,
    topk_values_ptr: *mut T,
    experts_indicator: *mut bool,
    indice_ptr: *mut bool,
    value_ptr: *mut T,
    topk_indices_ptr: *mut usize,
    token_index: usize,
    batch_size: usize,
    input_length: usize,



    
    output_length: usize,
) where
    T: Copy + PartialOrd + PartialEq + Default + AddAssign + Div<Output = T>,
{
    unsafe {
        let input_slice = std::slice::from_raw_parts(ptr1, input_length);
        let topk_len = output_length.min(input_length);

        let mut heap = FixedMinHeap::new(topk_values_ptr, topk_indices_ptr, topk_len);
        for (expert_idx, &value) in input_slice.iter().enumerate() {
            heap.push(value, expert_idx);
        }
        heap.sort_desc();
        let len = heap.len();

        let mut norm_sum = T::default();
        for k in 0..len {
            let expert_idx = *topk_indices_ptr.add(k);
            let value = *topk_values_ptr.add(k);
            norm_sum += value;
            *experts_indicator.add(expert_idx) = true;
        }
        let norm_sum = if norm_sum == T::default() {
            T::default()
        } else {
            norm_sum
        };

        for k in 0..len {
            let expert_idx = *topk_indices_ptr.add(k);
            let value = *topk_values_ptr.add(k);
            let prob = if norm_sum == T::default() {
                value
            } else {
                value / norm_sum
            };
            *topk_values_ptr.add(k) = prob;
            *topk_indices_ptr.add(k) = expert_idx;
            let offset = expert_idx * batch_size + token_index;
            *indice_ptr.add(offset) = true;
            *value_ptr.add(offset) = prob;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    #[test]
    fn test_experts_topk_norm_uses_heap_and_preserves_ties() {
        const NUM_EXPERTS: usize = 5;
        const NUM_TOPK: usize = 3;
        const BATCH_SIZE: usize = 2;
        const TOKEN_INDEX: usize = 1;

        let input = [0.2f32, 1.0, 1.0, 0.5, 1.0];
        let mut topk_values = [0.0f32; NUM_TOPK];
        let mut experts_indicator = [false; NUM_EXPERTS];
        let mut indices = [false; NUM_EXPERTS * BATCH_SIZE];
        let mut values = [0.0f32; NUM_EXPERTS * BATCH_SIZE];
        let mut topk_indices = [0usize; NUM_TOPK];

        unsafe {
            super::experts_topk_norm(
                input.as_ptr(),
                topk_values.as_mut_ptr(),
                experts_indicator.as_mut_ptr(),
                indices.as_mut_ptr(),
                values.as_mut_ptr(),
                topk_indices.as_mut_ptr(),
                TOKEN_INDEX,
                BATCH_SIZE,
                NUM_EXPERTS,
                NUM_TOPK,
            );
        }

        let expected_indices = [4usize, 2, 1];
        let expected_prob = 1.0f32 / 3.0f32;

        for (k, &idx) in expected_indices.iter().enumerate() {
            assert_eq!(topk_indices[k], idx);
            assert_relative_eq!(topk_values[k], expected_prob, epsilon = 1e-6);
            assert!(experts_indicator[idx]);
            let offset = idx * BATCH_SIZE + TOKEN_INDEX;
            assert!(indices[offset]);
            assert_relative_eq!(values[offset], expected_prob, epsilon = 1e-6);
        }

        for expert in 0..NUM_EXPERTS {
            if !expected_indices.contains(&expert) {
                assert!(!experts_indicator[expert]);
                let offset = expert * BATCH_SIZE + TOKEN_INDEX;
                assert!(!indices[offset]);
                assert_relative_eq!(values[offset], 0.0, epsilon = 1e-6);
            }
        }
    }
}
