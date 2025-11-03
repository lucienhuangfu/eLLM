use crate::kernel::generic::exp::Exp;
use crate::kernel::x86_64::f16_512::activation::exp512;
use std::ops::{AddAssign, Div, Sub};
use std::ptr;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;


// AVX-512 F16 Bitonic Sort implementation
#[inline(always)]
fn bitonic_sort_32_f16(values: __m512h, indices: __m512i) -> (__m512h, __m512i) {
    unsafe {
        let mut vals = values;
        let mut idxs = indices;
        
        // Stage 1: sort pairs (2 elements)
        (vals, idxs) = bitonic_merge_2_f16(vals, idxs);
        
        // Stage 2: sort groups of 4
        (vals, idxs) = bitonic_merge_4_f16(vals, idxs);
        
        // Stage 3: sort groups of 8
        (vals, idxs) = bitonic_merge_8_f16(vals, idxs);
        
        // Stage 4: sort groups of 16
        (vals, idxs) = bitonic_merge_16_f16(vals, idxs);
        
        // Stage 5: sort all 32 elements
        (vals, idxs) = bitonic_merge_32_f16(vals, idxs);
        
        (vals, idxs)
    }
}

#[inline(always)]
fn bitonic_merge_2_f16(values: __m512h, indices: __m512i) -> (__m512h, __m512i) {
    unsafe {
        // Create permutation for pairs: swap odd/even positions
        let perm = _mm512_set_epi16(
            30, 31, 28, 29, 26, 27, 24, 25,
            22, 23, 20, 21, 18, 19, 16, 17,
            14, 15, 12, 13, 10, 11, 8, 9,
            6, 7, 4, 5, 2, 3, 0, 1
        );
        
        let swapped_vals = _mm512_permutexvar_ph(perm, values);
        let swapped_idxs = _mm512_permutexvar_epi16(perm, indices);
        
        // Compare and select
        let mask = _mm512_cmp_ph_mask(values, swapped_vals, _CMP_GT_OQ);
        let sorted_vals = _mm512_mask_blend_ph(mask, values, swapped_vals);
        let sorted_idxs = _mm512_mask_blend_epi16(mask, indices, swapped_idxs);
        
        (sorted_vals, sorted_idxs)
    }
}

#[inline(always)]
fn bitonic_merge_4_f16(values: __m512h, indices: __m512i) -> (__m512h, __m512i) {
    unsafe {
        let mut vals = values;
        let mut idxs = indices;
        
        // First pass: compare distance 2
        let perm2 = _mm512_set_epi16(
            29, 28, 31, 30, 25, 24, 27, 26,
            21, 20, 23, 22, 17, 16, 19, 18,
            13, 12, 15, 14, 9, 8, 11, 10,
            5, 4, 7, 6, 1, 0, 3, 2
        );
        
        let swapped_vals = _mm512_permutexvar_ph(perm2, vals);
        let swapped_idxs = _mm512_permutexvar_epi16(perm2, idxs);
        
        // Create alternating mask for bitonic sequence
        let alt_mask = 0xCCCCCCCC_u32; // 11001100... pattern
        let cmp_mask = _mm512_cmp_ph_mask(vals, swapped_vals, _CMP_GT_OQ);
        let final_mask = cmp_mask ^ alt_mask;
        
        vals = _mm512_mask_blend_ph(final_mask, vals, swapped_vals);
        idxs = _mm512_mask_blend_epi16(final_mask, idxs, swapped_idxs);
        
        // Second pass: compare distance 1 (same as merge_2)
        bitonic_merge_2_f16(vals, idxs)
    }
}

#[inline(always)]
fn bitonic_merge_8_f16(values: __m512h, indices: __m512i) -> (__m512h, __m512i) {
    unsafe {
        let mut vals = values;
        let mut idxs = indices;
        
        // First pass: compare distance 4
        let perm4 = _mm512_set_epi16(
            27, 26, 25, 24, 31, 30, 29, 28,
            19, 18, 17, 16, 23, 22, 21, 20,
            11, 10, 9, 8, 15, 14, 13, 12,
            3, 2, 1, 0, 7, 6, 5, 4
        );
        
        let swapped_vals = _mm512_permutexvar_ph(perm4, vals);
        let swapped_idxs = _mm512_permutexvar_epi16(perm4, idxs);
        
        let alt_mask = 0xF0F0F0F0_u32; // 11110000... pattern
        let cmp_mask = _mm512_cmp_ph_mask(vals, swapped_vals, _CMP_GT_OQ);
        let final_mask = cmp_mask ^ alt_mask;
        
        vals = _mm512_mask_blend_ph(final_mask, vals, swapped_vals);
        idxs = _mm512_mask_blend_epi16(final_mask, idxs, swapped_idxs);
        
        // Subsequent passes
        (vals, idxs) = bitonic_merge_4_f16(vals, idxs);
        (vals, idxs)
    }
}

#[inline(always)]
fn bitonic_merge_16_f16(values: __m512h, indices: __m512i) -> (__m512h, __m512i) {
    unsafe {
        let mut vals = values;
        let mut idxs = indices;
        
        // First pass: compare distance 8
        let perm8 = _mm512_set_epi16(
            23, 22, 21, 20, 19, 18, 17, 16,
            31, 30, 29, 28, 27, 26, 25, 24,
            7, 6, 5, 4, 3, 2, 1, 0,
            15, 14, 13, 12, 11, 10, 9, 8
        );
        
        let swapped_vals = _mm512_permutexvar_ph(perm8, vals);
        let swapped_idxs = _mm512_permutexvar_epi16(perm8, idxs);
        
        let alt_mask = 0xFF00FF00_u32; // alternating 8-bit blocks
        let cmp_mask = _mm512_cmp_ph_mask(vals, swapped_vals, _CMP_GT_OQ);
        let final_mask = cmp_mask ^ alt_mask;
        
        vals = _mm512_mask_blend_ph(final_mask, vals, swapped_vals);
        idxs = _mm512_mask_blend_epi16(final_mask, idxs, swapped_idxs);
        
        // Subsequent passes
        (vals, idxs) = bitonic_merge_8_f16(vals, idxs);
        (vals, idxs)
    }
}

#[inline(always)]
fn bitonic_merge_32_f16(values: __m512h, indices: __m512i) -> (__m512h, __m512i) {
    unsafe {
        let mut vals = values;
        let mut idxs = indices;
        
        // First pass: compare distance 16
        let perm16 = _mm512_set_epi16(
            15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0,
            31, 30, 29, 28, 27, 26, 25, 24,
            23, 22, 21, 20, 19, 18, 17, 16
        );
        
        let swapped_vals = _mm512_permutexvar_ph(perm16, vals);
        let swapped_idxs = _mm512_permutexvar_epi16(perm16, idxs);
        
        let alt_mask = 0xFFFF0000_u32; // first 16 vs last 16
        let cmp_mask = _mm512_cmp_ph_mask(vals, swapped_vals, _CMP_GT_OQ);
        let final_mask = cmp_mask ^ alt_mask;
        
        vals = _mm512_mask_blend_ph(final_mask, vals, swapped_vals);
        idxs = _mm512_mask_blend_epi16(final_mask, idxs, swapped_idxs);
        
        // Subsequent passes
        (vals, idxs) = bitonic_merge_16_f16(vals, idxs);
        (vals, idxs)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use std::arch::x86_64::*;

    #[test]
    fn test_bitonic_sort_f16() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512fp16") {
            return;
        }
        
        unsafe {
            // Test data: reverse sorted array
            let test_data: [f16; 32] = [
                f16::from_f32(31.0), f16::from_f32(30.0), f16::from_f32(29.0), f16::from_f32(28.0),
                f16::from_f32(27.0), f16::from_f32(26.0), f16::from_f32(25.0), f16::from_f32(24.0),
                f16::from_f32(23.0), f16::from_f32(22.0), f16::from_f32(21.0), f16::from_f32(20.0),
                f16::from_f32(19.0), f16::from_f32(18.0), f16::from_f32(17.0), f16::from_f32(16.0),
                f16::from_f32(15.0), f16::from_f32(14.0), f16::from_f32(13.0), f16::from_f32(12.0),
                f16::from_f32(11.0), f16::from_f32(10.0), f16::from_f32(9.0), f16::from_f32(8.0),
                f16::from_f32(7.0), f16::from_f32(6.0), f16::from_f32(5.0), f16::from_f32(4.0),
                f16::from_f32(3.0), f16::from_f32(2.0), f16::from_f32(1.0), f16::from_f32(0.0),
            ];
            
            let values = _mm512_loadu_ph(test_data.as_ptr());
            let indices = _mm512_set_epi16(
                31, 30, 29, 28, 27, 26, 25, 24,
                23, 22, 21, 20, 19, 18, 17, 16,
                15, 14, 13, 12, 11, 10, 9, 8,
                7, 6, 5, 4, 3, 2, 1, 0
            );
            
            let (sorted_values, sorted_indices) = bitonic_sort_32_f16(values, indices);
            
            // Extract results
            let mut result_values: [f16; 32] = [f16::from_f32(0.0); 32];
            let mut result_indices: [i16; 32] = [0; 32];
            
            _mm512_storeu_ph(result_values.as_mut_ptr(), sorted_values);
            _mm512_storeu_epi16(result_indices.as_mut_ptr(), sorted_indices);
            
            // Verify sorting (descending order)
            for i in 1..32 {
                assert!(result_values[i-1].to_f32() >= result_values[i].to_f32());
            }
            
            // Verify that indices correspond to original positions
            assert_eq!(result_indices[0], 0);  // smallest value was at index 31, now at 0
            assert_eq!(result_indices[31], 31); // largest value was at index 0, now at 31
        }
    }
}