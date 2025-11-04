use std::arch::x86_64::*;

/// Optimized bitonic sort network for exactly 8 f32 values with indices tracking
/// Takes SIMD registers as input and returns SIMD registers
/// Now accepts custom indices as input
pub unsafe fn bitonic_sort_f32x8_with_indices(
    values: __m256,
    indices: __m256i,
) -> (__m256, __m256i) {
    let mut v = values;
    let mut idx = indices;

    // Stage 1: Sort pairs to create bitonic sequences
    (v, idx) = compare_swap_adjacent::<0b10110001>(v, idx); // swap (0,1), (2,3), (4,5), (6,7)

    // Stage 2: Sort groups of 4
    // First make bitonic: reverse order of pairs 1 and 3
    let reverse_mask = _mm256_set_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    v = _mm256_permutevar8x32_ps(v, reverse_mask);
    idx = _mm256_permutevar8x32_epi32(idx, reverse_mask);

    (v, idx) = compare_swap_distance(v, idx, _mm256_set_epi32(1, 0, 3, 2, 5, 4, 7, 6)); // swap (0,2), (1,3), (4,6), (5,7)
    (v, idx) = compare_swap_adjacent::<0b10110001>(v, idx); // swap (0,1), (2,3), (4,5), (6,7)

    // Stage 3: Sort all 8 elements
    // First make bitonic: reverse second half
    let reverse_mask = _mm256_set_epi32(0, 1, 2, 3, 7, 6, 5, 4);
    v = _mm256_permutevar8x32_ps(v, reverse_mask);
    idx = _mm256_permutevar8x32_epi32(idx, reverse_mask);

    (v, idx) = compare_swap_distance(v, idx, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4)); // swap (0,4), (1,5), (2,6), (3,7)
    (v, idx) = compare_swap_distance(v, idx, _mm256_set_epi32(1, 0, 3, 2, 5, 4, 7, 6)); // swap (0,2), (1,3), (4,6), (5,7)
    (v, idx) = compare_swap_adjacent::<0b10110001>(v, idx); // swap (0,1), (2,3), (4,5), (6,7)

    (v, idx)
}

/// Convenience function with default indices (0, 1, 2, 3, 4, 5, 6, 7)
pub unsafe fn bitonic_sort_f32x8_default_indices(values: __m256) -> (__m256, __m256i) {
    let default_indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    bitonic_sort_f32x8_with_indices(values, default_indices)
}

/// Compare and swap adjacent elements using shuffle pattern
unsafe fn compare_swap_adjacent<const SHUFFLE_PATTERN: i32>(
    v: __m256,
    idx: __m256i,
) -> (__m256, __m256i) {
    let swapped_v = _mm256_shuffle_ps(v, v, SHUFFLE_PATTERN);
    let swapped_idx = _mm256_shuffle_epi32(idx, SHUFFLE_PATTERN);

    let cmp = _mm256_cmp_ps(v, swapped_v, _CMP_GT_OQ);
    let cmp_int = _mm256_castps_si256(cmp);

    let result_v = _mm256_blendv_ps(v, swapped_v, cmp);
    let result_idx = _mm256_blendv_epi8(idx, swapped_idx, cmp_int);

    (result_v, result_idx)
}

/// Compare and swap elements using permutation mask
unsafe fn compare_swap_distance(
    v: __m256,
    idx: __m256i,
    permute_mask: __m256i,
) -> (__m256, __m256i) {
    let swapped_v = _mm256_permutevar8x32_ps(v, permute_mask);
    let swapped_idx = _mm256_permutevar8x32_epi32(idx, permute_mask);

    let cmp = _mm256_cmp_ps(v, swapped_v, _CMP_GT_OQ);
    let cmp_int = _mm256_castps_si256(cmp);

    let result_v = _mm256_blendv_ps(v, swapped_v, cmp);
    let result_idx = _mm256_blendv_epi8(idx, swapped_idx, cmp_int);

    (result_v, result_idx)
}

/// Helper function to create custom indices register from array
pub unsafe fn create_indices_from_array(indices: &[i32; 8]) -> __m256i {
    _mm256_set_epi32(
        indices[7], indices[6], indices[5], indices[4], indices[3], indices[2], indices[1],
        indices[0],
    )
}

/// Helper function to load f32 array into register
pub unsafe fn load_f32_array(values: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(values.as_ptr())
}

/// Helper function to store register to f32 array
pub unsafe fn store_f32_array(reg: __m256, values: &mut [f32; 8]) {
    _mm256_storeu_ps(values.as_mut_ptr(), reg);
}

/// Helper function to store indices register to i32 array
pub unsafe fn store_indices_array_i32(reg: __m256i, indices: &mut [i32; 8]) {
    _mm256_storeu_si256(indices.as_mut_ptr() as *mut __m256i, reg);
}

/// Helper function to store indices register to usize array
pub unsafe fn store_indices_array(reg: __m256i, indices: &mut [usize; 8]) {
    let mut temp = [0i32; 8];
    store_indices_array_i32(reg, &mut temp);
    for i in 0..8 {
        indices[i] = temp[i] as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitonic_sort_with_default_indices() {
        let values = [8.0, 3.0, 6.0, 1.0, 7.0, 2.0, 5.0, 4.0];

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [0, 4, 2, 6, 7, 1, 5, 3]);
        }
    }

    #[test]
    fn test_bitonic_sort_ascending_order() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // Already ascending

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            // Should sort to descending order
            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [7, 6, 5, 4, 3, 2, 1, 0]);
        }
    }

    #[test]
    fn test_bitonic_sort_descending_order() {
        let values = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]; // Already descending

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            // Should remain in descending order
            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [0, 1, 2, 3, 4, 5, 6, 7]);
        }
    }

    #[test]
    fn test_bitonic_sort_reverse_pairs() {
        let values = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]; // Pairs in reverse order

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [6, 7, 4, 5, 2, 3, 0, 1]);
        }
    }

    #[test]
    fn test_bitonic_sort_all_equal() {
        let values = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]; // All equal

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            // All values should remain the same
            assert_eq!(sorted_values, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
            // Indices order may vary but should be valid
            println!("All equal indices: {:?}", result_indices);
        }
    }

    #[test]
    fn test_bitonic_sort_with_duplicates() {
        let values = [5.0, 3.0, 5.0, 1.0, 3.0, 1.0, 5.0, 3.0];

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0]);
            println!("Duplicates indices: {:?}", result_indices);
        }
    }

    #[test]
    fn test_bitonic_sort_negative_values() {
        let values = [-1.0, 3.0, -5.0, 2.0, -3.0, 4.0, -2.0, 1.0];

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -5.0]);
            assert_eq!(result_indices, [5, 1, 3, 7, 0, 6, 4, 2]);
        }
    }

    #[test]
    fn test_bitonic_sort_alternating_pattern() {
        let values = [8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0]; // Alternating high-low

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [0, 2, 4, 6, 7, 5, 3, 1]);
        }
    }

    #[test]
    fn test_bitonic_sort_extreme_values() {
        let values = [
            f32::MAX,
            f32::MIN,
            0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            42.0,
        ];

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            // Check that infinity is first and negative infinity is last
            assert_eq!(sorted_values[0], f32::INFINITY);
            assert_eq!(sorted_values[7], f32::NEG_INFINITY);
            println!("Extreme values sorted: {:?}", sorted_values);
            println!("Extreme values indices: {:?}", result_indices);
        }
    }

    #[test]
    fn test_bitonic_sort_with_custom_indices() {
        let values = [8.0, 3.0, 6.0, 1.0, 7.0, 2.0, 5.0, 4.0];
        let custom_indices = [100, 101, 102, 103, 104, 105, 106, 107]; // Custom starting indices

        unsafe {
            let values_reg = load_f32_array(&values);
            let indices_reg = create_indices_from_array(&custom_indices);

            let (sorted_reg, indices_result_reg) =
                bitonic_sort_f32x8_with_indices(values_reg, indices_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0i32; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array_i32(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [100, 104, 102, 106, 107, 101, 105, 103]);
            // Custom indices preserved
        }
    }

    #[test]
    fn test_bitonic_sort_with_negative_indices() {
        let values = [8.0, 3.0, 6.0, 1.0, 7.0, 2.0, 5.0, 4.0];
        let custom_indices = [-1, -2, -3, -4, -5, -6, -7, -8]; // Negative indices

        unsafe {
            let values_reg = load_f32_array(&values);
            let indices_reg = create_indices_from_array(&custom_indices);

            let (sorted_reg, indices_result_reg) =
                bitonic_sort_f32x8_with_indices(values_reg, indices_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0i32; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array_i32(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [-1, -5, -3, -7, -8, -2, -6, -4]); // Negative indices preserved
        }
    }

    #[test]
    fn test_bitonic_sort_small_range() {
        let values = [2.1, 2.3, 2.0, 2.5, 2.2, 2.4, 2.6, 2.7]; // Small range values

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0]);
            assert_eq!(result_indices, [7, 6, 3, 5, 1, 4, 0, 2]);
        }
    }

    #[test]
    fn test_bitonic_sort_two_groups() {
        let values = [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]; // Two descending groups

        unsafe {
            let values_reg = load_f32_array(&values);
            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_default_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [4, 5, 6, 7, 0, 1, 2, 3]);
        }
    }
}
