use super::activation::exp256;
use crate::kernel::common::heap::FixedMinHeap;
use std::arch::x86_64::*;

use crate::kernel::x86_64::f32_256::bitonic_sort::bitonic_sort_f32x8_desc;

// Router 只保留 top-k 个专家，权重在选出的子集内归一化（HF 的
// `norm_topk_prob = true`）——本仓库唯一使用的语义，旧的二态开关已删。
pub fn experts_topk_softmax_norm(
    input_ptr: *const f32,
    topk_values_ptr: *mut f32,
    topk_indices_ptr: *mut usize,
    num_experts: usize,
    num_topk: usize,
) {
    unsafe {
        get_topk(
            input_ptr,
            topk_values_ptr,
            topk_indices_ptr,
            num_experts,
            num_topk,
        );
        softmax_topk_only_inplace(topk_values_ptr, num_topk);
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
    heap.sort_desc();
}

#[inline(always)]
unsafe fn softmax_topk_only_inplace(out_values: *mut f32, topk: usize) {
    debug_assert!(!out_values.is_null());
    debug_assert!(topk > 0 && topk <= 8);
    let max_val = *out_values;
    let mut lane_buf = [f32::NEG_INFINITY; 8];
    for i in 0..topk {
        lane_buf[i] = *out_values.add(i);
    }
    let values = _mm256_loadu_ps(lane_buf.as_ptr());
    let shifted = _mm256_sub_ps(values, _mm256_set1_ps(max_val));
    let exp_vals = exp256(shifted);
    _mm256_storeu_ps(lane_buf.as_mut_ptr(), exp_vals);
    let mut sum = 0.0f32;
    for i in 0..topk {
        sum += lane_buf[i];
    }
    let inv_sum = 1.0f32 / sum.max(f32::MIN_POSITIVE);
    for i in 0..topk {
        *out_values.add(i) = lane_buf[i] * inv_sum;
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

        let mut expected: Vec<(f32, usize)> = data
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, val)| (val, idx))
            .collect();
        expected.sort_by(|a, b| b.0.total_cmp(&a.0));
        for i in 0..(topk) {
            assert_ulps_eq!(out_vals[i], expected[i].0);
            assert_eq!(out_idx[i], expected[i].1);
        }
    }

    #[test]
    fn test_softmax_topk_only_inplace() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 子集内归一化：只拿 top-k 的值，max / denom 均在这几个数上算。
        let topk = 4;
        let original = [11.0f32, 10.0, 9.5, 8.0];
        let mut topk_vals = original;
        let max_val = original[0];
        let denom: f32 = original.iter().map(|v| (v - max_val).exp()).sum();
        unsafe {
            softmax_topk_only_inplace(topk_vals.as_mut_ptr(), topk);
        }
        for (i, v) in topk_vals.iter().enumerate() {
            let expected_prob = ((original[i] - max_val).exp()) / denom;
            assert_relative_eq!(*v, expected_prob, epsilon = 1e-3);
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
        unsafe {
            experts_topk_softmax_norm(
                data.as_ptr(),
                topk_vals.as_mut_ptr(),
                topk_idx.as_mut_ptr(),
                NUM_EXPERTS,
                NUM_TOPK,
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

        for i in 0..(NUM_TOPK) {
            let idx = expected[i].0;
            let prob = ((expected[i].1 - max_k).exp()) / denom_k;
            assert_eq!(topk_idx[i], idx);
            assert_relative_eq!(topk_vals[i], prob, epsilon = 1e-3);
        }
    }
}
