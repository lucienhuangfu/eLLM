#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

const LANES: i32 = 32;
const LANE_INDICES_I16: [i16; LANES as usize] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

#[inline(always)]
unsafe fn lane_index_vector_i16() -> __m512i {
    _mm512_loadu_si512(LANE_INDICES_I16.as_ptr() as *const _)
}

#[target_feature(enable = "avx512f,avx512bw,avx512fp16")]
pub(crate) unsafe fn bitonic_sort_f16x32_desc(
    mut values: __m512h,
    mut indices: __m512i,
) -> (__m512h, __m512i) {
    let lane_idx = lane_index_vector_i16();
    let zero = _mm512_setzero_si512();
    let mut k = 2;
    while k <= LANES {
        let k_vec = _mm512_set1_epi16(k as i16);
        let stage_mask = _mm512_cmpeq_epi16_mask(_mm512_and_si512(lane_idx, k_vec), zero);
        let mut j = k / 2;
        loop {
            let j_vec = _mm512_set1_epi16(j as i16);
            let perm_idx = _mm512_xor_si512(lane_idx, j_vec);
            let perm_vals = _mm512_permutexvar_ph(perm_idx, values);
            let perm_ids = _mm512_permutexvar_epi16(perm_idx, indices);
            let gt_mask = _mm512_cmp_ph_mask(values, perm_vals, _CMP_GT_OQ);
            let hi_vals = _mm512_mask_blend_ph(gt_mask, perm_vals, values);
            let lo_vals = _mm512_mask_blend_ph(gt_mask, values, perm_vals);
            let hi_ids = _mm512_mask_blend_epi16(gt_mask, perm_ids, indices);
            let lo_ids = _mm512_mask_blend_epi16(gt_mask, indices, perm_ids);
            values = _mm512_mask_blend_ph(stage_mask, lo_vals, hi_vals);
            indices = _mm512_mask_blend_epi16(stage_mask, lo_ids, hi_ids);
            if j == 1 {
                break;
            }
            j /= 2;
        }
        k <<= 1;
    }
    (values, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::arch::is_x86_feature_detected;

    #[test]
    fn bitonic_sort_desc_orders_f16_values_and_indices() {
        if !(is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512fp16"))
        {
            eprintln!("Skipping AVX-512 FP16-dependent test");
            return;
        }
        let input_vals = [
            1.0f32, -5.5, 3.0, 7.25, 0.5, 12.0, -1.25, 4.75, 9.5, -3.75, 2.5, 6.5, 8.0, -2.0, 5.5,
            10.25, 11.5, -4.5, 13.0, 14.75, 15.5, -6.0, 16.25, 17.5, 18.75, -7.5, 19.25, 20.5,
            21.75, -8.25, 22.0, 23.5,
        ];
        let mut value_bits = [0u16; LANES as usize];
        for (dst, &val) in value_bits.iter_mut().zip(input_vals.iter()) {
            *dst = f16::from_f32(val).to_bits();
        }
        let mut expected_pairs: Vec<(f32, i16)> = value_bits
            .iter()
            .enumerate()
            .map(|(i, &bits)| (f16::from_bits(bits).to_f32(), i as i16))
            .collect();
        expected_pairs
            .sort_by(|(av, ai), (bv, bi)| bv.partial_cmp(av).unwrap().then_with(|| ai.cmp(bi)));
        unsafe {
            let values_vec = _mm512_loadu_ph(value_bits.as_ptr() as *const _);
            let mut idx_arr = [0i16; LANES as usize];
            for (i, slot) in idx_arr.iter_mut().enumerate() {
                *slot = i as i16;
            }
            let indices_vec = _mm512_loadu_si512(idx_arr.as_ptr() as *const _);
            let (sorted_vals, sorted_ids) = bitonic_sort_f16x32_desc(values_vec, indices_vec);
            let mut sorted_vals_bits = [0u16; LANES as usize];
            _mm512_storeu_ph(sorted_vals_bits.as_mut_ptr() as *mut _, sorted_vals);
            let mut sorted_vals_f32 = [0f32; LANES as usize];
            for (dst, &bits) in sorted_vals_f32.iter_mut().zip(sorted_vals_bits.iter()) {
                *dst = f16::from_bits(bits).to_f32();
            }
            let mut sorted_idx = [0i16; LANES as usize];
            _mm512_storeu_si512(sorted_idx.as_mut_ptr() as *mut _, sorted_ids);
            let expected_vals: Vec<f32> = expected_pairs.iter().map(|(v, _)| *v).collect();
            let expected_idx: Vec<i16> = expected_pairs.iter().map(|(_, i)| *i).collect();
            assert_eq!(sorted_vals_f32.to_vec(), expected_vals);
            assert_eq!(sorted_idx.iter().copied().collect::<Vec<_>>(), expected_idx);
        }
    }
}
