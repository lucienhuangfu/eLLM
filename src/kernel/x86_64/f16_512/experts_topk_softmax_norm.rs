use crate::kernel::generic::exp::Exp;
use crate::kernel::x86_64::f16_512::activation::exp512;
use std::ops::{AddAssign, Div, Sub};
use std::ptr;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub fn experts_topk_softmax_norm(
    input_ptr: *const f16,
    // [num_experts]
    experts_indicator_ptr: *mut bool,
    // token_size = sequence_chunk_size * batch_size
    // [num_experts, token_size]
    output_indices_ptr: *mut bool,
    // [num_experts, token_size]
    output_values_ptr: *mut f16,
    index_token: usize,
    num_token: usize,
    num_experts: usize,
    num_topk: usize,
) {
    unsafe {
        // Use AVX-512 F16 registers as heap: 32 f16 values per register
        let mut heap_values = _mm512_setzero_ph(); // 32 f16 values
        let mut heap_indices = _mm512_setzero_epi16(); // 32 u16 indices

        // Initialize heap with first num_topk elements
        let mut init_vals = [f16::ZERO; 32];
        let mut init_indices = [0u16; 32];

        for idx in 0..num_topk.min(32).min(num_experts) {
            init_vals[idx] = *input_ptr.add(idx);
            init_indices[idx] = idx;
        }

        heap_values = _mm512_loadu_ph(init_vals.as_ptr());
        heap_indices = _mm512_loadu_epi16(init_indices.as_ptr());

        // Build min-heap using AVX-512 F16 sorting
        simd_heapify_f16(&mut heap_values, &mut heap_indices, num_topk);

        // Process remaining elements using AVX-512 F16
        let mut idx = num_topk;
        while idx < num_experts {
            // Load 32 f16 values
            let vals = _mm512_loadu_ph(input_ptr.add(idx));
            let indices_vec = _mm512_setr_epi16(
                (idx + 0),
                (idx + 1),
                (idx + 2),
                (idx + 3),
                (idx + 4),
                (idx + 5),
                (idx + 6),
                (idx + 7),
                (idx + 8),
                (idx + 9),
                (idx + 10),
                (idx + 11),
                (idx + 12),
                (idx + 13),
                (idx + 14),
                (idx + 15),
                (idx + 16),
                (idx + 17),
                (idx + 18),
                (idx + 19),
                (idx + 20),
                (idx + 21),
                (idx + 22),
                (idx + 23),
                (idx + 24),
                (idx + 25),
                (idx + 26),
                (idx + 27),
                (idx + 28),
                (idx + 29),
                (idx + 30),
                (idx + 31),
            );

            // Get heap minimum (broadcast first element)
            let min_val = _mm512_broadcastw_epi16(_mm512_extracti32x4_epi32(heap_indices, 0));
            let min_val_f16 = _mm512_broadcastw_epi16(_mm512_castph_si512(heap_values));

            // Compare with heap minimum using native f16 comparison
            let cmp_mask = _mm512_cmp_ph_mask(vals, _mm512_castsi512_ph(min_val_f16), _CMP_GT_OQ);

            // Update heap with new candidates
            simd_heap_update_f16(
                &mut heap_values,
                &mut heap_indices,
                vals,
                indices_vec,
                cmp_mask,
                num_topk,
            );

            idx += 32;
        }

        // Sort heap to get descending order
        simd_heap_sort_f16(&mut heap_values, &mut heap_indices, num_topk);

        // Calculate sum using AVX-512 F16 operations for normalization
        let mut sum_accum = _mm512_setzero_ph();
        let mut i = 0;

        while i < num_experts {
            let vals = _mm512_loadu_ph(input_ptr.add(i));
            let exp_vals = exp512(vals);
            sum_accum = _mm512_add_ph(sum_accum, exp_vals);
            i += 32;
        }

        // Horizontal sum of f16 values
        let sum_f16 = horizontal_sum_f16_native(sum_accum);

        // Apply softmax directly to heap_values (top-k values)
        let exp_heap_values = exp512(heap_values);
        let sum_broadcast = _mm512_set1_ph(sum_f16.to_bits());
        let softmax_values = _mm512_div_ph(exp_heap_values, sum_broadcast);

        // Extract results and set outputs
        let mut final_vals = [f16::ZERO; 32];
        let mut final_indices = [0u16; 32];
        _mm512_storeu_ph(final_vals.as_mut_ptr(), softmax_values);
        _mm512_storeu_epi16(final_indices.as_mut_ptr(), heap_indices);

        for k in 0..num_topk {
            let expert_idx = final_indices[k] as usize;
            let softmax_value = final_vals[k];
            *experts_indicator_ptr.add(expert_idx) = true;
            *output_indices_ptr.add(expert_idx * num_token + index_token) = true;
            *(output_values_ptr.add(expert_idx * num_token + index_token)) = softmax_value;
        }
    }
}

#[inline(always)]
unsafe fn simd_heapify_f16(values: &mut __m512h, indices: &mut __m512i, heap_size: usize) {
    // Bitonic sort network for small heap_size
    for step in 0..5 {
        // log2(32) = 5
        let stride = 1 << step;
        for i in (0..heap_size).step_by(stride * 2) {
            let end = (i + stride * 2).min(heap_size);

            // Compare and swap within stride
            for j in i..(i + stride).min(end) {
                if j + stride < end {
                    let val1 = _mm512_extractf32x4_ps(_mm512_castph_ps(*values), j / 4);
                    let val2 = _mm512_extractf32x4_ps(_mm512_castph_ps(*values), (j + stride) / 4);

                    // Use f16 comparison
                    let cmp = _mm512_cmp_ph_mask(
                        _mm512_castps_ph(val1),
                        _mm512_castps_ph(val2),
                        _CMP_GT_OQ,
                    );

                    if (cmp & (1 << (j % 4))) != 0 {
                        // Swap values and indices
                        let temp_val = _mm512_extractf32x4_ps(_mm512_castph_ps(*values), j / 4);
                        let temp_idx = _mm512_extracti32x4_epi32(*indices, j / 8);

                        *values = _mm512_insertf32x4(*values, val2, j / 4);
                        *values = _mm512_insertf32x4(*values, temp_val, (j + stride) / 4);

                        *indices = _mm512_inserti32x4(
                            *indices,
                            _mm512_extracti32x4_epi32(*indices, (j + stride) / 8),
                            j / 8,
                        );
                        *indices = _mm512_inserti32x4(*indices, temp_idx, (j + stride) / 8);
                    }
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn simd_heap_update_f16(
    heap_values: &mut __m512h,
    heap_indices: &mut __m512i,
    new_values: __m512h,
    new_indices: __m512i,
    mask: __mmask32,
    heap_size: usize,
) {
    // Process each bit in mask
    for i in 0..32 {
        if (mask & (1 << i)) != 0 {
            // Replace heap minimum (first element)
            *heap_values = _mm512_mask_broadcastw_epi16(
                *heap_values,
                1,
                _mm512_extracti32x4_epi32(_mm512_castph_si512(new_values), i / 8),
            );
            *heap_indices = _mm512_mask_broadcastw_epi16(
                *heap_indices,
                1,
                _mm512_extracti32x4_epi32(new_indices, i / 8),
            );

            // Re-heapify
            simd_heapify_f16(heap_values, heap_indices, heap_size);
        }
    }
}

#[inline(always)]
unsafe fn simd_heap_sort_f16(values: &mut __m512h, indices: &mut __m512i, heap_size: usize) {
    // Selection sort using f16 comparisons
    for i in 0..heap_size {
        let mut max_idx = i;
        let base_val = _mm512_extractf32x4_ps(_mm512_castph_ps(*values), i / 4);

        for j in (i + 1)..heap_size {
            let comp_val = _mm512_extractf32x4_ps(_mm512_castph_ps(*values), j / 4);
            let cmp = _mm512_cmp_ph_mask(
                _mm512_castps_ph(comp_val),
                _mm512_castps_ph(base_val),
                _CMP_GT_OQ,
            );

            if (cmp & (1 << (j % 4))) != 0 {
                max_idx = j;
            }
        }

        if max_idx != i {
            // Swap using f16 operations
            let val_i = _mm512_extractf32x4_ps(_mm512_castph_ps(*values), i / 4);
            let val_max = _mm512_extractf32x4_ps(_mm512_castph_ps(*values), max_idx / 4);
            let idx_i = _mm512_extracti32x4_epi32(*indices, i / 8);
            let idx_max = _mm512_extracti32x4_epi32(*indices, max_idx / 8);

            *values = _mm512_insertf32x4(*values, val_max, i / 4);
            *values = _mm512_insertf32x4(*values, val_i, max_idx / 4);
            *indices = _mm512_inserti32x4(*indices, idx_max, i / 8);
            *indices = _mm512_inserti32x4(*indices, idx_i, max_idx / 8);
        }
    }
}

#[inline(always)]
unsafe fn horizontal_sum_f16_native(x: __m512h) -> f16 {
    // Manual reduction since _mm512_reduce_add_ph may not be available
    let mut result = [f16::ZERO; 32];
    _mm512_storeu_ph(result.as_mut_ptr(), x);

    let mut sum = 0.0f32;
    for i in 0..32 {
        sum += result[i].to_f32();
    }
    f16::from_f32(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use std::arch::x86_64::*;

    #[test]
    fn test_horizontal_sum_f16_native() {
        unsafe {
            // Test case 1: All ones
            let ones = _mm512_set1_ph(f16::from_f32(1.0).to_bits());
            let sum = horizontal_sum_f16_native(ones);
            assert_ulps_eq!(sum.to_f32(), 32.0, max_ulps = 1);

            // Test case 2: Sequential values 1-32
            let mut vals = [f16::ZERO; 32];
            for i in 0..32 {
                vals[i] = f16::from_f32(i as f32 + 1.0);
            }
            let vec = _mm512_loadu_ph(vals.as_ptr());
            let sum = horizontal_sum_f16_native(vec);
            let expected = (1..=32).sum::<i32>() as f32; // 528
            assert_ulps_eq!(sum.to_f32(), expected, max_ulps = 2);

            // Test case 3: Zeros
            let zeros = _mm512_setzero_ph();
            let sum = horizontal_sum_f16_native(zeros);
            assert_eq!(sum.to_f32(), 0.0);
        }
    }

    #[test]
    fn test_simd_heapify_f16() {
        unsafe {
            // Test case 1: Simple 4-element heap
            let mut vals = [f16::ZERO; 32];
            vals[0] = f16::from_f32(3.0);
            vals[1] = f16::from_f32(1.0);
            vals[2] = f16::from_f32(4.0);
            vals[3] = f16::from_f32(2.0);

            let mut indices = [0u16; 32];
            indices[0] = 0;
            indices[1] = 1;
            indices[2] = 2;
            indices[3] = 3;

            let mut heap_values = _mm512_loadu_ph(vals.as_ptr());
            let mut heap_indices = _mm512_loadu_epi16(indices.as_ptr());

            simd_heapify_f16(&mut heap_values, &mut heap_indices, 4);

            _mm512_storeu_ph(vals.as_mut_ptr(), heap_values);
            _mm512_storeu_epi16(indices.as_mut_ptr(), heap_indices);

            // After heapify, minimum should be at root (for min-heap)
            assert!(vals[0] <= vals[1] && vals[0] <= vals[2] && vals[0] <= vals[3]);
        }
    }

    #[test]
    fn test_simd_heap_sort_f16() {
        unsafe {
            // Test case: Sort 8 random values in descending order
            let mut vals = [f16::ZERO; 32];
            let test_values = [5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0];
            let mut indices = [0u16; 32];

            for i in 0..8 {
                vals[i] = f16::from_f32(test_values[i]);
                indices[i] = i as u16;
            }

            let mut heap_values = _mm512_loadu_ph(vals.as_ptr());
            let mut heap_indices = _mm512_loadu_epi16(indices.as_ptr());

            simd_heap_sort_f16(&mut heap_values, &mut heap_indices, 8);

            _mm512_storeu_ph(vals.as_mut_ptr(), heap_values);
            _mm512_storeu_epi16(indices.as_mut_ptr(), heap_indices);

            // Check descending order
            for i in 0..7 {
                assert!(
                    vals[i] >= vals[i + 1],
                    "Values should be in descending order"
                );
            }

            // Check that largest value is first
            assert_ulps_eq!(vals[0].to_f32(), 9.0, max_ulps = 1);
        }
    }

    #[test]
    fn test_simd_heap_update_f16() {
        unsafe {
            // Initialize heap with small values
            let mut vals = [f16::ZERO; 32];
            vals[0] = f16::from_f32(1.0); // minimum
            vals[1] = f16::from_f32(2.0);
            vals[2] = f16::from_f32(3.0);

            let mut indices = [0u16; 32];
            indices[0] = 0;
            indices[1] = 1;
            indices[2] = 2;

            let mut heap_values = _mm512_loadu_ph(vals.as_ptr());
            let mut heap_indices = _mm512_loadu_epi16(indices.as_ptr());

            // New larger values to add
            let mut new_vals = [f16::ZERO; 32];
            new_vals[0] = f16::from_f32(5.0); // Should replace minimum
            let mut new_indices_arr = [0u16; 32];
            new_indices_arr[0] = 10;

            let new_values = _mm512_loadu_ph(new_vals.as_ptr());
            let new_indices = _mm512_loadu_epi16(new_indices_arr.as_ptr());
            let mask = 0x1; // Only first element

            simd_heap_update_f16(
                &mut heap_values,
                &mut heap_indices,
                new_values,
                new_indices,
                mask,
                3,
            );

            _mm512_storeu_ph(vals.as_mut_ptr(), heap_values);
            _mm512_storeu_epi16(indices.as_mut_ptr(), heap_indices);

            // The minimum (1.0) should be replaced by 5.0
            let contains_five = (0..3).any(|i| (vals[i].to_f32() - 5.0).abs() < 0.1);
            assert!(contains_five, "Heap should contain the new value 5.0");
        }
    }

    #[test]
    fn test_experts_topk_softmax_norm_edge_cases() {
        // Test case 1: All equal values
        let num_experts = 64;
        let num_topk = 4;
        let num_token = 16;
        let index_token = 3;

        let input_f16: Vec<f16> = vec![f16::from_f32(2.0); num_experts];
        let mut experts_indicator = vec![false; num_experts];
        let mut indices = vec![false; num_experts * num_token];
        let mut values = vec![f16::ZERO; num_experts * num_token];

        experts_topk_softmax_norm(
            input_f16.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indices.as_mut_ptr(),
            values.as_mut_ptr(),
            index_token,
            num_token,
            num_experts,
            num_topk,
        );

        // Should select exactly top-k experts
        let true_count = experts_indicator.iter().filter(|&&x| x).count();
        assert_eq!(true_count, num_topk);

        // All selected experts should have equal softmax values
        let mut selected_values = Vec::new();
        for expert_idx in 0..num_experts {
            if experts_indicator[expert_idx] {
                selected_values.push(values[expert_idx * num_token + index_token].to_f32());
            }
        }

        for i in 1..selected_values.len() {
            assert_ulps_eq!(selected_values[0], selected_values[i], max_ulps = 5);
        }
    }

    #[test]
    fn test_experts_topk_softmax_norm_small_input() {
        // Test case: Minimum viable input (32 experts, 1 top-k)
        let num_experts = 32;
        let num_topk = 1;
        let num_token = 8;
        let index_token = 2;

        let mut input = vec![f16::from_f32(1.0); num_experts];
        input[15] = f16::from_f32(10.0); // Make one clearly largest

        let mut experts_indicator = vec![false; num_experts];
        let mut indices = vec![false; num_experts * num_token];
        let mut values = vec![f16::ZERO; num_experts * num_token];

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

        // Only expert 15 should be selected
        assert!(experts_indicator[15]);
        assert_eq!(experts_indicator.iter().filter(|&&x| x).count(), 1);

        // Expert 15 should have non-zero value
        let value_idx = 15 * num_token + index_token;
        assert!(values[value_idx] > f16::ZERO);

        // All other experts should have zero values
        for expert_idx in 0..num_experts {
            if expert_idx != 15 {
                let value_idx = expert_idx * num_token + index_token;
                assert_eq!(values[value_idx], f16::ZERO);
            }
        }
    }

    #[test]
    fn test_experts_topk_softmax_norm_large_input() {
        // Test case: Larger input to stress test
        let num_experts = 256; // 8 * 32
        let num_topk = 16;
        let num_token = 64;
        let index_token = 10;

        let input: Vec<f16> = (0..num_experts)
            .map(|i| f16::from_f32((i as f32 * 0.05) % 8.0))
            .collect();

        let mut experts_indicator = vec![false; num_experts];
        let mut indices = vec![false; num_experts * num_token];
        let mut values = vec![f16::ZERO; num_experts * num_token];

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

        // Should select exactly top-k experts
        let true_count = experts_indicator.iter().filter(|&&x| x).count();
        assert_eq!(true_count, num_topk);

        // Sum of softmax values for selected experts should be reasonable
        let mut sum_values = 0.0f32;
        for expert_idx in 0..num_experts {
            if experts_indicator[expert_idx] {
                sum_values += values[expert_idx * num_token + index_token].to_f32();
            }
        }
        assert!(sum_values > 0.0 && sum_values < 1.0);
    }

    #[test]
    fn test_experts_topk_softmax_norm() {
        // Original comprehensive test
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
        let mut values = vec![0.0f16; num_experts * num_token];

        let input_f16: Vec<f16> = input.iter().map(|&x| f16::from_f32(x)).collect();

        experts_topk_softmax_norm(
            input_f16.as_ptr(),
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
            sum_values += values[expert_idx * num_token + index_token].to_f32();
        }
        assert!(sum_values > 0.0 && sum_values < 1.0);

        // Check that values are in descending order by checking actual positions
        for i in 0..num_topk - 1 {
            let val1 = values[expected_experts[i] * num_token + index_token].to_f32();
            let val2 = values[expected_experts[i + 1] * num_token + index_token].to_f32();
            assert!(val1 >= val2, "Values should be in descending order");
        }

        // Check that non-selected experts have zero values
        for expert_idx in 0..num_experts {
            if !expected_experts.contains(&expert_idx) {
                let value_idx = expert_idx * num_token + index_token;
                assert_eq!(
                    values[value_idx].to_f32(),
                    0.0,
                    "Non-selected expert {} should have zero value",
                    expert_idx
                );
            }
        }
    }
}
