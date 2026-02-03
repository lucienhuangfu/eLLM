use crate::kernel::common::heap::FixedMinHeap;
use crate::kernel::x86_64::f16_512::activation::exp512;
use crate::kernel::x86_64::f16_512::bitonic_sort::bitonic_sort_f16_desc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::f16;

const LANES: usize = 32;

#[inline(always)]
unsafe fn sum_f16_vec(vec: __m512h, count: usize) -> f32 {
    debug_assert!(count == LANES);

    // Convert to f32 for reduction to maintain precision
    let vec_i = _mm512_castph_si512(vec);
    let lower = _mm512_cvtph_ps(_mm512_castsi512_si256(vec_i));
    let upper = _mm512_cvtph_ps(_mm512_extracti64x4_epi64::<1>(vec_i));

    _mm512_reduce_add_ps(_mm512_add_ps(lower, upper))
}

#[inline(always)]
// #[target_feature(enable = "avx512f,avx512bw,avx512fp16")]
pub unsafe fn get_topk(
    in_ptr: *const f16,
    out_values: *mut f16,
    out_indices: *mut usize,
    len: usize,
    topk: usize,
) {
    debug_assert!(!in_ptr.is_null());
    debug_assert!(!out_values.is_null());
    debug_assert!(!out_indices.is_null());
    debug_assert!(len % LANES == 0 && len > 0, "len must be divisible by 32");
    debug_assert!(topk > 0 && topk <= LANES, "topk must be within 1..=32");
    debug_assert!(topk <= len, "topk cannot exceed len");

    let mut heap = FixedMinHeap::new(out_values, out_indices, topk);
    let mut chunk_start = 0usize;

    // Buffers for sorting a single chunk
    let mut chunk_vals = [0.0f16; LANES];
    let mut chunk_idx = [0i16; LANES];

    while chunk_start < len {
        let values = _mm512_loadu_ph(in_ptr.add(chunk_start) as *const _);
        _mm512_storeu_ph(chunk_vals.as_mut_ptr() as *mut _, values);

        // Initialize indices for this chunk (0..31)
        for i in 0..LANES {
            chunk_idx[i] = i as i16;
        }

        // Sort this chunk
        bitonic_sort_f16_desc(chunk_vals.as_mut_ptr(), chunk_idx.as_mut_ptr(), LANES);

        // Push top k from this chunk to heap
        let chunk_take = topk.min(LANES);
        for i in 0..chunk_take {
            let val = chunk_vals[i];
            let local_idx = chunk_idx[i];
            let global_idx = chunk_start + local_idx as usize;
            heap.push(val, global_idx);
        }

        chunk_start += LANES;
    }

    /*
    // Sort the result in descending order of values
    let mut pairs: Vec<(f16, usize)> = (0..topk)
        .map(|i| (*out_values.add(i), *out_indices.add(i)))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for i in 0..topk {
        *out_values.add(i) = pairs[i].0;
        *out_indices.add(i) = pairs[i].1;
    }*/
}

#[inline(always)]
// #[target_feature(enable = "avx512f,avx512bw,avx512fp16")]
unsafe fn softmax_stats_avx512_fp16(input_ptr: *const f16, len: usize) -> (f32, f32) {
    debug_assert!(!input_ptr.is_null());
    debug_assert!(len % LANES == 0 && len > 0);

    let mut max_vec = _mm512_loadu_ph(input_ptr as *const _);
    let mut offset = LANES;
    while offset < len {
        let vals = _mm512_loadu_ph(input_ptr.add(offset) as *const _);
        max_vec = _mm512_max_ph(max_vec, vals);
        offset += LANES;
    }

    let mut tmp = [0.0f16; LANES];
    _mm512_storeu_ph(tmp.as_mut_ptr() as *mut _, max_vec);
    let mut max_scalar = f16::NEG_INFINITY;
    for &v in &tmp {
        max_scalar = max_scalar.max(v);
    }
    let mut max_scalar_f32 = max_scalar as f32;
    if !max_scalar_f32.is_finite() {
        max_scalar_f32 = 0.0;
        max_scalar = 0.0;
    }

    let max_broadcast = _mm512_set1_ph(max_scalar);
    let mut denom = 0.0f32;
    offset = 0;
    while offset < len {
        let vals = _mm512_loadu_ph(input_ptr.add(offset) as *const _);
        let exp_vals = exp512(_mm512_sub_ph(vals, max_broadcast));
        denom += sum_f16_vec(exp_vals, LANES);
        offset += LANES;
    }

    denom = denom.max(f32::MIN_POSITIVE);
    (max_scalar_f32, denom)
}

#[inline(always)]
unsafe fn softmax_topk_inplace(out_values: *mut f16, topk: usize, max_val: f32, denom: f32) {
    debug_assert!(!out_values.is_null());
    debug_assert!(topk > 0 && topk <= LANES);

    let mut lane_buf = [0.0f16; LANES];
    for i in 0..topk {
        lane_buf[i] = *out_values.add(i);
    }

    let values = _mm512_loadu_ph(lane_buf.as_ptr() as *const _);
    let shifted = _mm512_sub_ph(values, _mm512_set1_ph(max_val as f16));
    let exp_vals = exp512(shifted);
    let scale = _mm512_set1_ph((1.0f32 / denom.max(f32::MIN_POSITIVE)) as f16);
    let normalized = _mm512_mul_ph(exp_vals, scale);
    _mm512_storeu_ph(lane_buf.as_mut_ptr() as *mut _, normalized);

    for i in 0..topk {
        *out_values.add(i) = lane_buf[i];
    }
}

pub fn experts_topk_softmax_norm(
    input_ptr: *const f16,
    topk_values_ptr: *mut f16,
    topk_indices_ptr: *mut usize,
    experts_indicator_ptr: *mut bool,
    indices_ptr: *mut bool,
    value_ptr: *mut f16,
    index_token: usize,
    num_token: usize,
    num_experts: usize,
    num_topk: usize,
    norm_topk_prob: bool,
) {
    debug_assert!(!input_ptr.is_null());
    debug_assert!(!topk_values_ptr.is_null());
    debug_assert!(!topk_indices_ptr.is_null());
    debug_assert!(!experts_indicator_ptr.is_null());
    debug_assert!(!indices_ptr.is_null());
    debug_assert!(!value_ptr.is_null());
    debug_assert!(num_topk > 0 && num_topk <= num_experts);
    debug_assert!(index_token < num_token);

    unsafe {
        get_topk(
            input_ptr,
            topk_values_ptr,
            topk_indices_ptr,
            num_experts,
            num_topk,
        );
        let (max_val, denom) = softmax_stats_avx512_fp16(input_ptr, num_experts);
        softmax_topk_inplace(topk_values_ptr, num_topk, max_val, denom);

        if norm_topk_prob {
            let mut sum = 0.0;
            for i in 0..num_topk {
                sum += *topk_values_ptr.add(i);
            }
            let scale = sum.recip();
            for i in 0..num_topk {
                *topk_values_ptr.add(i) *= scale;
            }
        }

        for i in 0..num_topk {
            let expert_idx = *topk_indices_ptr.add(i);
            *experts_indicator_ptr.add(expert_idx) = true;
            let offset = expert_idx * num_token + index_token;
            *indices_ptr.add(offset) = true;
            *value_ptr.add(offset) = *topk_values_ptr.add(i);
        }
        std::slice::from_raw_parts_mut(topk_indices_ptr, num_topk).sort_unstable();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_ulps_eq};
    use std::f16;

    #[test]
    fn test_get_topk() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512fp16") {
            return;
        }

        let data: [f16; 32] = [
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
        ];
        let topk = 4;
        let mut out_vals = [0.0; 4];
        let mut out_idx = [0usize; 4];

        unsafe {
            get_topk(
                data.as_ptr(),
                out_vals.as_mut_ptr(),
                out_idx.as_mut_ptr(),
                data.len(),
                topk,
            );
        }

        let mut result_pairs: Vec<(f32, usize)> = (0..topk)
            .map(|i| ((out_vals[i] as f32), out_idx[i]))
            .collect();
        // Sort descending to compare with expected
        result_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut expected: Vec<(f32, usize)> = data
            .iter()
            .enumerate()
            .map(|(idx, &val)| ((val as f32), idx))
            .collect();
        expected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        for i in 0..topk {
            assert_relative_eq!(result_pairs[i].0, expected[i].0, epsilon = 1e-3);
            assert_eq!(result_pairs[i].1, expected[i].1);
        }
    }

    #[test]
    fn test_softmax_stats_avx512_fp16() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512fp16") {
            return;
        }
        let data: [f16; 32] = [
            -0.5, 0.25, 3.75, -2.0, 6.0, 1.75, -4.25, 2.5, 0.0, 5.25, -1.25, 4.0, 3.0, -3.5, 7.5,
            2.25, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
        ];

        let data_f32: Vec<f32> = data.iter().map(|&x| (x as f32)).collect();
        let expected_max = data_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let expected_denom: f32 = data_f32.iter().map(|v| (v - expected_max).exp()).sum();

        let (max_val, denom) = unsafe { softmax_stats_avx512_fp16(data.as_ptr(), data.len()) };

        assert_relative_eq!(max_val, expected_max, epsilon = 1e-2);
        assert_relative_eq!(denom, expected_denom, max_relative = 0.05, epsilon = 1e-3);
    }

    #[test]
    fn test_softmax_topk_inplace() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512fp16") {
            return;
        }
        let topk = 4;
        let mut topk_vals: [f16; 4] = [10.0, 9.5, 8.0, 7.5];

        let max_val = 10.0f32;
        let data_f32: Vec<f32> = topk_vals.iter().map(|&x| (x as f32)).collect();
        let denom: f32 = data_f32.iter().map(|v| (v - max_val).exp()).sum();

        unsafe {
            softmax_topk_inplace(topk_vals.as_mut_ptr(), topk, max_val, denom);
        }

        for i in 0..topk {
            let expected = (data_f32[i] - max_val).exp() / denom;
            assert_relative_eq!((topk_vals[i] as f32), expected, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_experts_topk_softmax_norm() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512fp16") {
            return;
        }
        const NUM_EXPERTS: usize = 32;
        const NUM_TOPK: usize = 4;
        const NUM_TOKEN: usize = 3;
        const INDEX_TOKEN: usize = 1;

        let mut data = [0.0f16; NUM_EXPERTS];
        for i in 0..NUM_EXPERTS {
            let val = (i as f32) * 0.5 * if i % 2 == 0 { 1.0 } else { -1.0 };
            data[i] = val as f16;
        }
        data[5] = 10.0;
        data[10] = 9.0;
        data[15] = 8.0;
        data[20] = 7.0;

        let mut topk_vals = [0.0; NUM_TOPK];
        let mut topk_idx = [0usize; NUM_TOPK];
        let mut expert_flags = [false; NUM_EXPERTS];
        let mut indices = [false; NUM_EXPERTS * NUM_TOKEN];
        let mut values = [0.0; NUM_EXPERTS * NUM_TOKEN];

        unsafe {
            experts_topk_softmax_norm(
                data.as_ptr(),
                topk_vals.as_mut_ptr(),
                topk_idx.as_mut_ptr(),
                expert_flags.as_mut_ptr(),
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
            .enumerate()
            .map(|(idx, &val)| (idx, (val as f32)))
            .collect();
        expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k_subset: Vec<f32> = expected.iter().take(NUM_TOPK).map(|x| x.1).collect();
        let max_global = expected
            .iter()
            .map(|x| x.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let denom_global: f32 = expected.iter().map(|x| (x.1 - max_global).exp()).sum();

        let mut top_k_probs: Vec<f32> = top_k_subset
            .iter()
            .map(|&v| (v - max_global).exp() / denom_global)
            .collect();

        // Since we passed true for norm_topk_prob, we expect normalized probabilities
        let sum_probs: f32 = top_k_probs.iter().sum();
        for p in &mut top_k_probs {
            *p /= sum_probs;
        }

        let mut result_indices = Vec::new();
        for i in 0..NUM_TOPK {
            result_indices.push(topk_idx[i]);
        }
        let mut expected_indices: Vec<usize> =
            expected.iter().take(NUM_TOPK).map(|x| x.0).collect();
        expected_indices.sort_unstable();
        assert_eq!(result_indices, expected_indices);

        for (rank, (expert_idx, _)) in expected.iter().take(NUM_TOPK).enumerate() {
            let expected_prob = top_k_probs[rank];
            assert!(expert_flags[*expert_idx]);
            let offset = expert_idx * NUM_TOKEN + INDEX_TOKEN;
            assert!(indices[offset]);
            let actual_val = (values[offset] as f32);
            assert_relative_eq!(actual_val, expected_prob, epsilon = 1e-2);
        }
    }
}
