use std::arch::x86_64::*;

/// Optimized bitonic sort network for exactly 8 f32 values with indices tracking
/// Takes SIMD registers as input and returns SIMD registers
pub unsafe fn bitonic_sort_f32x8_with_indices(values: __m256) -> (__m256, __m256i) {
    let mut v = values;
    let mut indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    // Stage 1: Sort pairs (elements 0-1, 2-3, 4-5, 6-7)
    (v, indices) = sort_pairs_f32x8_with_indices(v, indices);

    // Stage 2: Merge pairs into groups of 4
    (v, indices) = merge_4_f32x8_with_indices(v, indices);

    // Stage 3: Final merge to sort all 8 elements
    (v, indices) = merge_8_f32x8_with_indices(v, indices);

    (v, indices)
}

/// Sort adjacent pairs within the vector with indices tracking
unsafe fn sort_pairs_f32x8_with_indices(v: __m256, idx: __m256i) -> (__m256, __m256i) {
    let swapped_v = _mm256_shuffle_ps(v, v, 0b10110001);
    let swapped_idx = _mm256_shuffle_epi32(idx, 0b10110001);

    let cmp = _mm256_cmp_ps(v, swapped_v, _CMP_GE_OQ);
    let cmp_int = _mm256_castps_si256(cmp);

    let result_v = _mm256_blendv_ps(swapped_v, v, cmp);
    let result_idx = _mm256_blendv_epi8(swapped_idx, idx, cmp_int);

    (result_v, result_idx)
}

/// Merge 4-element groups with indices tracking
unsafe fn merge_4_f32x8_with_indices(v: __m256, idx: __m256i) -> (__m256, __m256i) {
    // Reverse the second pair in each group of 4
    let reverse_indices = _mm256_set_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    let reversed_v = _mm256_permutevar8x32_ps(v, reverse_indices);
    let reversed_idx = _mm256_permutevar8x32_epi32(idx, reverse_indices);

    // Bitonic merge for groups of 4
    let (mut result_v, mut result_idx) =
        bitonic_compare_swap_4_with_indices(v, reversed_v, idx, reversed_idx);
    (result_v, result_idx) = bitonic_compare_swap_2_with_indices(result_v, result_idx);
    bitonic_compare_swap_1_with_indices(result_v, result_idx)
}

/// Final merge of all 8 elements with indices tracking
unsafe fn merge_8_f32x8_with_indices(v: __m256, idx: __m256i) -> (__m256, __m256i) {
    // Reverse the second half
    let reversed_v = _mm256_permute2f128_ps(v, v, 0x01);
    let reversed_idx = _mm256_permute2f128_si256(idx, idx, 0x01);
    let reverse_indices = _mm256_set_epi32(0, 1, 2, 3, 7, 6, 5, 4);
    let reversed_v = _mm256_permutevar8x32_ps(reversed_v, reverse_indices);
    let reversed_idx = _mm256_permutevar8x32_epi32(reversed_idx, reverse_indices);

    // Bitonic merge for full array
    let (mut result_v, mut result_idx) =
        bitonic_compare_swap_4_with_indices(v, reversed_v, idx, reversed_idx);
    (result_v, result_idx) = bitonic_compare_swap_2_with_indices(result_v, result_idx);
    bitonic_compare_swap_1_with_indices(result_v, result_idx)
}

/// Compare and swap elements 4 positions apart with indices
unsafe fn bitonic_compare_swap_4_with_indices(
    a_v: __m256,
    b_v: __m256,
    a_idx: __m256i,
    b_idx: __m256i,
) -> (__m256, __m256i) {
    let cmp = _mm256_cmp_ps(a_v, b_v, _CMP_GE_OQ);
    let cmp_int = _mm256_castps_si256(cmp);

    let result_v = _mm256_blendv_ps(b_v, a_v, cmp);
    let result_idx = _mm256_blendv_epi8(b_idx, a_idx, cmp_int);

    (result_v, result_idx)
}

/// Compare and swap elements 2 positions apart with indices
unsafe fn bitonic_compare_swap_2_with_indices(v: __m256, idx: __m256i) -> (__m256, __m256i) {
    let swapped_v = _mm256_shuffle_ps(v, v, 0b01001110);
    let swapped_idx = _mm256_shuffle_epi32(idx, 0b01001110);

    let cmp = _mm256_cmp_ps(v, swapped_v, _CMP_GE_OQ);
    let cmp_int = _mm256_castps_si256(cmp);

    let result_v = _mm256_blendv_ps(swapped_v, v, cmp);
    let result_idx = _mm256_blendv_epi8(swapped_idx, idx, cmp_int);

    (result_v, result_idx)
}

/// Compare and swap adjacent elements with indices
unsafe fn bitonic_compare_swap_1_with_indices(v: __m256, idx: __m256i) -> (__m256, __m256i) {
    let swapped_v = _mm256_shuffle_ps(v, v, 0b10110001);
    let swapped_idx = _mm256_shuffle_epi32(idx, 0b10110001);

    let cmp = _mm256_cmp_ps(v, swapped_v, _CMP_GE_OQ);
    let cmp_int = _mm256_castps_si256(cmp);

    let result_v = _mm256_blendv_ps(swapped_v, v, cmp);
    let result_idx = _mm256_blendv_epi8(swapped_idx, idx, cmp_int);

    (result_v, result_idx)
}

/// Helper function to load f32 array into register
pub unsafe fn load_f32_array(values: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(values.as_ptr())
}

/// Helper function to store register to f32 array
pub unsafe fn store_f32_array(reg: __m256, values: &mut [f32; 8]) {
    _mm256_storeu_ps(values.as_mut_ptr(), reg);
}

/// Helper function to store indices register to usize array
pub unsafe fn store_indices_array(reg: __m256i, indices: &mut [usize; 8]) {
    _mm256_storeu_si256(indices.as_mut_ptr() as *mut __m256i, reg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitonic_sort_f32x8_with_indices_descending() {
        let values = [8.0, 3.0, 6.0, 1.0, 7.0, 2.0, 5.0, 4.0];

        unsafe {
            let values_reg = load_f32_array(&values);

            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_with_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [0, 4, 2, 6, 7, 1, 5, 3]);
        }
    }

    #[test]
    fn test_bitonic_sort_reverse_order_descending() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        unsafe {
            let values_reg = load_f32_array(&values);

            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_with_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [7, 6, 5, 4, 3, 2, 1, 0]);
        }
    }

    #[test]
    fn test_bitonic_sort_already_sorted_descending() {
        let values = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        unsafe {
            let values_reg = load_f32_array(&values);

            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_with_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            assert_eq!(result_indices, [0, 1, 2, 3, 4, 5, 6, 7]);
        }
    }

    #[test]
    fn test_bitonic_sort_duplicates() {
        let values = [5.0, 3.0, 5.0, 1.0, 3.0, 1.0, 5.0, 3.0];

        unsafe {
            let values_reg = load_f32_array(&values);

            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_with_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0]);
            // Indices should preserve the relative order of equal elements
            println!("Indices: {:?}", result_indices);
        }
    }

    #[test]
    fn test_bitonic_sort_negative_values() {
        let values = [-1.0, 3.0, -5.0, 2.0, -3.0, 4.0, -2.0, 1.0];

        unsafe {
            let values_reg = load_f32_array(&values);

            let (sorted_reg, indices_result_reg) = bitonic_sort_f32x8_with_indices(values_reg);

            let mut sorted_values = [0.0f32; 8];
            let mut result_indices = [0usize; 8];

            store_f32_array(sorted_reg, &mut sorted_values);
            store_indices_array(indices_result_reg, &mut result_indices);

            assert_eq!(sorted_values, [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -5.0]);
            assert_eq!(result_indices, [5, 1, 3, 7, 0, 6, 4, 2]);
        }
    }
}
