use std::arch::x86_64::*;
use std::f16;

const LANES: i32 = 32;
#[repr(align(64))]
struct AlignedLaneIndices([i16; LANES as usize]);

static LANE_INDICES_I16: AlignedLaneIndices = AlignedLaneIndices([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
]);
unsafe fn lane_index_vector_i16() -> __m512i {
    _mm512_loadu_si512(LANE_INDICES_I16.0.as_ptr() as *const _)
}

#[target_feature(enable = "avx512f,avx512bw,avx512fp16")]
pub(crate) unsafe fn bitonic_sort_f16_desc(values: *mut f16, indices: *mut i16, len: usize) {
    assert!(len % 32 == 0);
    let lane_idx = lane_index_vector_i16();
    let zero = _mm512_setzero_si512();
    let mut offset = 0;
    while offset < len {
        let v_ptr = values.add(offset);
        let i_ptr = indices.add(offset);
        let mut v = _mm512_loadu_ph(v_ptr as *const _);
        let mut i = _mm512_loadu_si512(i_ptr as *const _);
        let mut k = 2;
        while k <= LANES {
            let k_vec = _mm512_set1_epi16(k as i16);
            let stage_mask = _mm512_cmpeq_epi16_mask(_mm512_and_si512(lane_idx, k_vec), zero);
            let mut j = k / 2;
            loop {
                let j_vec = _mm512_set1_epi16(j as i16);
                let perm_idx = _mm512_xor_si512(lane_idx, j_vec);
                let perm_vals = _mm512_permutexvar_ph(perm_idx, v);
                let perm_ids = _mm512_permutexvar_epi16(perm_idx, i);
                let gt_mask = _mm512_cmp_ph_mask(v, perm_vals, _CMP_GT_OQ);
                let hi_vals = _mm512_mask_blend_ph(gt_mask, perm_vals, v);
                let lo_vals = _mm512_mask_blend_ph(gt_mask, v, perm_vals);
                let hi_ids = _mm512_mask_blend_epi16(gt_mask, perm_ids, i);
                let lo_ids = _mm512_mask_blend_epi16(gt_mask, i, perm_ids);

                let is_upper = _mm512_test_epi16_mask(lane_idx, j_vec);
                let sort_mask = stage_mask ^ is_upper;

                v = _mm512_mask_blend_ph(sort_mask, lo_vals, hi_vals);
                i = _mm512_mask_blend_epi16(sort_mask, lo_ids, hi_ids);
                if j == 1 {
                    break;
                }
                j /= 2;
            }
            k <<= 1;
        }
        _mm512_storeu_ph(v_ptr as *mut _, v);
        _mm512_storeu_si512(i_ptr as *mut _, i);
        offset += 32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::is_x86_feature_detected;
    use std::f16;

    #[test]
    fn bitonic_sort_desc_orders_f16_values_and_indices() {
        if !(is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512fp16"))
        {
            eprintln!("Skipping AVX-512 FP16-dependent test");
            return;
        }
        let input_vals_32: [f16; LANES as usize] = [
            1.0, -5.5, 3.0, 7.25, 0.5, 12.0, -1.25, 4.75, 9.5, -3.75, 2.5, 6.5, 8.0, -2.0, 5.5,
            10.25, 11.5, -4.5, 13.0, 14.75, 15.5, -6.0, 16.25, 17.5, 18.75, -7.5, 19.25, 20.5,
            21.75, -8.25, 22.0, 23.5,
        ];
        let mut input_vals = Vec::new();
        input_vals.extend_from_slice(&input_vals_32);
        input_vals.extend_from_slice(&input_vals_32);

        let mut indices: Vec<i16> = (0..64).collect();

        let mut expected_pairs_1: Vec<(f16, i16)> = input_vals_32
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i as i16))
            .collect();
        expected_pairs_1
            .sort_by(|(av, ai), (bv, bi)| bv.partial_cmp(av).unwrap().then_with(|| ai.cmp(bi)));

        let mut expected_pairs_2: Vec<(f16, i16)> = input_vals_32
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, (i + 32) as i16))
            .collect();
        expected_pairs_2
            .sort_by(|(av, ai), (bv, bi)| bv.partial_cmp(av).unwrap().then_with(|| ai.cmp(bi)));

        unsafe {
            bitonic_sort_f16_desc(input_vals.as_mut_ptr(), indices.as_mut_ptr(), 64);
        }

        let expected_vals: Vec<f16> = expected_pairs_1
            .iter()
            .map(|(v, _)| *v)
            .chain(expected_pairs_2.iter().map(|(v, _)| *v))
            .collect();
        let expected_idx: Vec<i16> = expected_pairs_1
            .iter()
            .map(|(_, i)| *i)
            .chain(expected_pairs_2.iter().map(|(_, i)| *i))
            .collect();

        assert_eq!(input_vals, expected_vals);
        assert_eq!(indices, expected_idx);
    }
}
