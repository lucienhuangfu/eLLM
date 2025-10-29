use crate::kernel::generic::exp::Exp;
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
    indices_ptr: *mut bool,
    // [num_experts, token_size]
    value_ptr: *mut f16,
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
        heap_indices = _mm512_loadu_epi16(initIndices.as_ptr());

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
                (idx + 7) ,
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

        // Calculate sum using AVX-512 F16 operations
        let mut sum_accum = _mm512_setzero_ph();
        let mut i = 0;

        while i < num_experts {
            let vals = _mm512_loadu_ph(input_ptr.add(i));
            // Native f16 exp approximation
            let exp_vals = simd_exp_f16_native(vals);
            sum_accum = _mm512_add_ph(sum_accum, exp_vals);
            i += 32;
        }

        // Horizontal sum of f16 values
        let sum_f16 = horizontal_sum_f16_native(sum_accum);

        // Extract results and set outputs
        let mut final_vals = [f16::ZERO; 32];
        let mut final_indices = [0u16; 32];
        _mm512_storeu_ph(final_vals.as_mut_ptr(), heap_values);
        _mm512_storeu_epi16(final_indices.as_mut_ptr(), heap_indices);

        for k in 0..num_topk {
            let expert_idx = final_indices[k] as usize;
            let val_f16 = final_vals[k];

            *experts_indicator_ptr.add(expert_idx) = true;
            *indices_ptr.add(expert_idx * num_token + index_token) = true;

            // Native f16 operations
            let exp_val = scalar_exp_f16_native(val_f16);
            let softmax_value = _mm512_div_ph(
                _mm512_set1_ph(exp_val.to_bits()),
                _mm512_set1_ph(sum_f16.to_bits()),
            );
            let softmax_scalar = _mm512_cvtsh_h(softmax_value);

            *(value_ptr.add(expert_idx * num_token + index_token)) = softmax_scalar;
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
unsafe fn simd_exp_f16_native(x: __m512h) -> __m512h {
    // Polynomial approximation using native f16 operations
    let one = _mm512_set1_ph(1.0f16.to_bits());
    let half = _mm512_set1_ph(0.5f16.to_bits());
    let sixth = _mm512_set1_ph((1.0f32 / 6.0).to_bits() as u16); // Approximate

    let x2 = _mm512_mul_ph(x, x);
    let x3 = _mm512_mul_ph(x2, x);

    let term2 = _mm512_mul_ph(x2, half);
    let term3 = _mm512_mul_ph(x3, sixth);

    let result = _mm512_add_ph(one, x);
    let result = _mm512_add_ph(result, term2);
    _mm512_add_ph(result, term3)
}

#[inline(always)]
unsafe fn horizontal_sum_f16_native(x: __m512h) -> f16 {
    // Use native f16 reduction
    let reduced = _mm512_reduce_add_ph(x);
    f16::from_bits(reduced)
}

#[inline(always)]
fn scalar_exp_f16_native(x: f16) -> f16 {
    let x_f32 = x.to_f32();
    if x_f32 > 10.0 {
        return f16::from_f32(22026.0);
    }
    if x_f32 < -10.0 {
        return f16::from_f32(0.0000454);
    }

    let result = 1.0 + x_f32 + x_f32 * x_f32 * 0.5 + x_f32 * x_f32 * x_f32 * (1.0 / 6.0);
    f16::from_f32(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_experts_topk_softmax_norm() {
        // Test data: 128 experts (divisible by 32)
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
