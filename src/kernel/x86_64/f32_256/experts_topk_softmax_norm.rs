// use crate::num_traits::Exp;

use num_traits::Inv;

use super::activation::exp256;
use crate::kernel::common::heap::FixedMinHeap;
use std::arch::x86_64::*;

use crate::kernel::x86_64::f32_256::bitonic_sort::bitonic_sort_f32x8_desc;

pub fn experts_topk_softmax_norm(
    input_ptr: *const f32,
    topk_values_ptr: *mut f32,
    topk_indices_ptr: *mut usize,
    // [num_experts]
    experts_indicator_ptr: *mut bool,
    // [num_experts, batch_size]
    indices_ptr: *mut bool,
    value_ptr: *mut f32,
    index_token: usize,
    num_token: usize,
    num_experts: usize,
    num_topk: usize,
    norm_topk_prob: bool,
) {
    unsafe {
        get_topk(
            input_ptr,
            topk_values_ptr,
            topk_indices_ptr,
            num_experts,
            num_topk,
        );
        let (max_val, denom) = softmax_stats_avx(input_ptr, num_experts);
        softmax_topk_inplace(topk_values_ptr, num_topk, max_val, denom);

        if norm_topk_prob == true {
            // Normalize top-k probabilities to sum to 1
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
            let offset = (expert_idx) * (num_token) + (index_token);
            *indices_ptr.add(offset) = true;
            *value_ptr.add(offset) = *topk_values_ptr.add(i);
        }

        std::slice::from_raw_parts_mut(topk_indices_ptr, num_topk).sort_unstable();
    }
}

#[inline(always)]
pub unsafe fn get_topk(
    in_ptr: *const f32,
    out_values: *mut f32,
    out_indices: *mut usize,
    len: usize,
    topk: usize,
) {
    debug_assert!(!in_ptr.is_null());
    debug_assert!(!out_values.is_null());
    debug_assert!(!out_indices.is_null());
    debug_assert!(len % 8 == 0, "len must be divisible by 8");
    debug_assert!(topk > 0 && topk <= 8, "topk must be within 1..=8");
    debug_assert!(topk <= len, "topk cannot exceed len");

    let lane_offsets = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut heap = FixedMinHeap::new(out_values, out_indices, topk);

    for chunk_start in (0..len).step_by(8) {
        let values = _mm256_loadu_ps(in_ptr.add(chunk_start));
        let base = _mm256_set1_epi32(chunk_start as i32);
        let indices = _mm256_add_epi32(base, lane_offsets);
        let (sorted_vals, sorted_idx) = bitonic_sort_f32x8_desc(values, indices);

        let mut chunk_vals = [0.0f32; 8];
        let mut chunk_idx = [0i32; 8];
        _mm256_storeu_ps(chunk_vals.as_mut_ptr(), sorted_vals);
        _mm256_storeu_si256(chunk_idx.as_mut_ptr() as *mut __m256i, sorted_idx);

        let chunk_take = topk.min(8);
        for lane in 0..(chunk_take) {
            heap.push(chunk_vals[lane], chunk_idx[lane] as usize);
        }
    }
    debug_assert_eq!(heap.len(), topk);
    // heap.sort_desc();
}

#[inline(always)]
unsafe fn softmax_stats_avx(input_ptr: *const f32, len: usize) -> (f32, f32) {
    debug_assert!(!input_ptr.is_null());
    debug_assert!(len % 8 == 0 && len > 0);
    let mut max_vec = _mm256_loadu_ps(input_ptr);
    let mut offset = 8;
    while offset < len {
        let vals = _mm256_loadu_ps(input_ptr.add(offset));
        max_vec = _mm256_max_ps(max_vec, vals);
        offset += 8;
    }
    let mut tmp = _mm256_permute2f128_ps(max_vec, max_vec, 0x01);
    max_vec = _mm256_max_ps(max_vec, tmp);
    tmp = _mm256_shuffle_ps(max_vec, max_vec, 0x4E);
    max_vec = _mm256_max_ps(max_vec, tmp);
    tmp = _mm256_shuffle_ps(max_vec, max_vec, 0xB1);
    max_vec = _mm256_max_ps(max_vec, tmp);
    let max_scalar = _mm_cvtss_f32(_mm256_castps256_ps128(max_vec));
    let max_broadcast = _mm256_set1_ps(max_scalar);

    let mut sum_vec = _mm256_setzero_ps();
    let mut sum_offset = 0;
    while sum_offset < len {
        let vals = _mm256_loadu_ps(input_ptr.add(sum_offset));
        let shifted = _mm256_sub_ps(vals, max_broadcast);
        let exp_vals = exp256(shifted);
        sum_vec = _mm256_add_ps(sum_vec, exp_vals);
        sum_offset += 8;
    }
    let mut acc = _mm256_hadd_ps(sum_vec, sum_vec);
    acc = _mm256_hadd_ps(acc, acc);
    let swapped = _mm256_permute2f128_ps(acc, acc, 0x01);
    let total = _mm256_add_ps(acc, swapped);
    let mut denom = _mm_cvtss_f32(_mm256_castps256_ps128(total));
    denom = denom.max(f32::MIN_POSITIVE);
    (max_scalar, denom)
}

#[inline(always)]
unsafe fn softmax_topk_inplace(out_values: *mut f32, topk: usize, max_val: f32, denom: f32) {
    debug_assert!(!out_values.is_null());
    debug_assert!(topk > 0 && topk <= 8);
    let effective_denom = denom.max(f32::MIN_POSITIVE);
    let mut lane_buf = [f32::NEG_INFINITY; 8];
    for i in 0..topk {
        lane_buf[i] = *out_values.add(i);
    }
    let values = _mm256_loadu_ps(lane_buf.as_ptr());
    let shifted = _mm256_sub_ps(values, _mm256_set1_ps(max_val));
    let exp_vals = exp256(shifted);
    let normalized = _mm256_mul_ps(exp_vals, _mm256_set1_ps(1.0 / effective_denom));
    _mm256_storeu_ps(lane_buf.as_mut_ptr(), normalized);
    for i in 0..topk {
        *out_values.add(i) = lane_buf[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_ulps_eq};

    #[test]
    fn test_get_topk() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let data = [
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        ];
        let topk = 4;
        let mut out_vals = [0.0f32; 4];
        let mut out_idx = [0; 4];
        unsafe {
            get_topk(
                data.as_ptr(),
                out_vals.as_mut_ptr(),
                out_idx.as_mut_ptr(),
                data.len(),
                topk,
            );
        }
        let raw_vals = out_vals;
        unsafe {
            let (max_val, denom) = softmax_stats_avx(data.as_ptr(), data.len());
            softmax_topk_inplace(out_vals.as_mut_ptr(), topk, max_val, denom);
        }

        let mut expected: Vec<(f32, usize)> = data
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, val)| (val, idx))
            .collect();
        expected.sort_by(|a, b| b.0.total_cmp(&a.0));
        for i in 0..(topk) {
            assert_ulps_eq!(raw_vals[i], expected[i].0);
            assert_eq!(out_idx[i], expected[i].1);
        }
        let max_val = expected[0].0;
        let denom = data.iter().map(|v| (v - max_val).exp()).sum::<f32>();
        for i in 0..(topk) {
            let expected_prob = ((expected[i].0 - max_val).exp()) / denom;
            assert_relative_eq!(out_vals[i], expected_prob, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_softmax_stats_avx() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let data = [
            -0.5, 0.25, 3.75, -2.0, 6.0, 1.75, -4.25, 2.5, 0.0, 5.25, -1.25, 4.0, 3.0, -3.5, 7.5,
            2.25,
        ];
        let expected_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let expected_denom: f32 = data.iter().map(|v| (v - expected_max).exp()).sum();
        let (max_val, denom) = unsafe { softmax_stats_avx(data.as_ptr(), data.len()) };
        assert_relative_eq!(max_val, expected_max, epsilon = 1e-6);
        assert_relative_eq!(denom, expected_denom, epsilon = 1e-3);
    }

    #[test]
    fn test_softmax_topk_inplace() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let data = [
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        ];
        let topk = 4;
        let mut expected: Vec<(f32, usize)> = data
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, val)| (val, idx))
            .collect();
        expected.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut topk_vals = [expected[0].0, expected[1].0, expected[2].0, expected[3].0];
        let (max_val, denom) = unsafe { softmax_stats_avx(data.as_ptr(), data.len()) };
        unsafe {
            softmax_topk_inplace(topk_vals.as_mut_ptr(), topk, max_val, denom);
        }
        let norm = denom;
        for i in 0..(topk) {
            let expected_prob = ((expected[i].0 - max_val).exp()) / norm;
            assert_relative_eq!(topk_vals[i], expected_prob, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_experts_topk_softmax_norm() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        const NUM_EXPERTS: usize = 16;
        const NUM_TOPK: usize = 4;
        const NUM_TOKEN: usize = 3;
        const INDEX_TOKEN: usize = 1;
        let data = [
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        ];
        let mut topk_vals = [0.0f32; NUM_TOPK];
        let mut topk_idx = [0; NUM_TOPK];
        let mut expert_flags = [false; NUM_EXPERTS];
        let mut indices = [false; (NUM_EXPERTS * NUM_TOKEN)];
        let mut values = [0.0f32; (NUM_EXPERTS * NUM_TOKEN)];

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
            .copied()
            .enumerate()
            .map(|(idx, val)| (idx, val))
            .collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1));

        let expected_topk_vals: Vec<f32> = expected.iter().take(NUM_TOPK).map(|x| x.1).collect();
        let max_k = expected_topk_vals
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let denom_k: f32 = expected_topk_vals.iter().map(|v| (v - max_k).exp()).sum();

        let mut is_topk = [false; NUM_EXPERTS];

        // Verify indices are sorted
        let mut expected_indices: Vec<usize> =
            expected.iter().take(NUM_TOPK).map(|x| x.0).collect();
        expected_indices.sort_unstable();
        for i in 0..(NUM_TOPK) {
            assert_eq!(topk_idx[i], expected_indices[i]);
        }

        for i in 0..(NUM_TOPK) {
            let idx = expected[i].0;
            let prob = ((expected[i].1 - max_k).exp()) / denom_k;

            // Check sparse outputs
            assert!(expert_flags[idx]);
            let offset = (idx * NUM_TOKEN + INDEX_TOKEN);
            assert!(indices[offset]);
            assert_relative_eq!(values[offset], prob, epsilon = 1e-3);
            is_topk[idx] = true;
        }

        for expert in 0..(NUM_EXPERTS) {
            if !is_topk[expert] {
                assert!(!expert_flags[expert]);
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
}





