use crate::kernel::common::heap::FixedMinHeap;
use std::arch::x86_64::*;
use std::ptr;

use crate::kernel::x86_64::f32_256::bitonic_sort::bitonic_sort_f32x8_desc;

pub fn topk_softmax(
    // [thread_num, topk_size]
    input_values_ptr: *const f32,
    // [thread_num, topk_size]
    input_indices_ptr: *const usize,
    // [thread_num]
    sums_ptr: *const f32,
    // [topk_size]
    output_values_ptr: *mut f32,
    // [topk_size]
    output_indices_ptr: *mut usize,
    // [1]
    output_token_ptr: *mut usize,
    thread_num: usize,
    topk_size: usize,
) {
    unsafe {
        // get topk from input
        get_topk_with_indices(
            input_values_ptr,
            input_indices_ptr,
            output_values_ptr,
            output_indices_ptr,
            thread_num * topk_size,
            topk_size,
        );

        // Get max value directly from first element (sort results are ordered)
        let max_val = *output_values_ptr.add(0);
        // Calculate adjusted total sum (subtract max for numerical stability)
        let mut total_sum = 0.0;
        for i in 0..thread_num {
            total_sum +=
                (*sums_ptr.add(i)) * (*input_values_ptr.add(i * topk_size) - max_val).exp();
        }

        total_sum -= (max_val * ((topk_size * thread_num) as f32));

        // Normalize using the adjusted total sum
        for i in 0..topk_size {
            let val = *output_values_ptr.add(i);
            let exp_val = (val - max_val).exp();
            let normalized_val = exp_val / total_sum;
            ptr::write(output_values_ptr.add(i), normalized_val);
        }
        ptr::write(output_token_ptr, *output_indices_ptr);
    }
}

#[inline(always)]
pub unsafe fn get_topk_with_indices(
    in_ptr: *const f32,
    in_indices: *const usize,
    out_values: *mut f32,
    out_indices: *mut usize,
    len: usize,
    topk: usize,
) {
    debug_assert!(!in_ptr.is_null());
    debug_assert!(!in_indices.is_null());
    debug_assert!(!out_values.is_null());
    debug_assert!(!out_indices.is_null());
    debug_assert!(len % 8 == 0, "len must be divisible by 8");
    debug_assert!(topk > 0 && topk <= 8, "topk must be within 1..=8");
    debug_assert!(topk <= len, "topk cannot exceed len");

    let mut heap = FixedMinHeap::new(out_values, out_indices, topk);

    for chunk_start in (0..len).step_by(8) {
        let values = _mm256_loadu_ps(in_ptr.add(chunk_start));
        let mut chunk_idx_usize = [0usize; 8];
        ptr::copy_nonoverlapping(in_indices.add(chunk_start), chunk_idx_usize.as_mut_ptr(), 8);
        let mut idx_i32 = [0i32; 8];
        for lane in 0..8 {
            debug_assert!(chunk_idx_usize[lane] <= i32::MAX as usize);
            idx_i32[lane] = chunk_idx_usize[lane] as i32;
        }
        let idx_vec = _mm256_loadu_si256(idx_i32.as_ptr() as *const __m256i);
        let (sorted_vals, sorted_idx) = bitonic_sort_f32x8_desc(values, idx_vec);

        let mut chunk_vals = [0.0f32; 8];
        let mut sorted_idx_i32 = [0i32; 8];
        _mm256_storeu_ps(chunk_vals.as_mut_ptr(), sorted_vals);
        _mm256_storeu_si256(sorted_idx_i32.as_mut_ptr() as *mut __m256i, sorted_idx);

        let chunk_take = topk.min(8);
        for lane in 0..chunk_take {
            heap.push(chunk_vals[lane], sorted_idx_i32[lane] as usize);
        }
    }
    debug_assert_eq!(heap.len(), topk);
    heap.sort_desc();
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_get_topk_with_indices() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let data = [
            1.5, -0.75, 4.25, 2.0, 5.5, 3.5, -1.25, 6.75, 0.0, 2.75, 7.25, -3.0, 8.5, 1.25, 9.0,
            -2.5,
        ];
        let indices: [usize; 16] = [10, 21, 5, 42, 7, 18, 33, 2, 44, 13, 8, 29, 1, 55, 3, 60];
        let topk = 4;
        let mut out_vals = [0.0f32; 4];
        let mut out_idx = [0usize; 4];

        unsafe {
            get_topk_with_indices(
                data.as_ptr(),
                indices.as_ptr(),
                out_vals.as_mut_ptr(),
                out_idx.as_mut_ptr(),
                data.len(),
                topk,
            );
        }

        let mut expected: Vec<(f32, usize)> =
            data.iter().copied().zip(indices.iter().copied()).collect();
        expected.sort_by(|a, b| b.0.total_cmp(&a.0));

        for i in 0..topk {
            assert_ulps_eq!(out_vals[i], expected[i].0);
            assert_eq!(out_idx[i], expected[i].1);
        }
    }

    #[test]
    fn test_topk_softmax() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let topk_size = 8usize;
        let thread_num = 4usize;
        let values: [f32; 32] = [
            5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4,
            3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9,
        ];
        let indices: Vec<usize> = (0..topk_size * thread_num).collect();
        let global_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let share_per_thread = global_max * ((topk_size * thread_num) as f32) / (thread_num as f32);
        let mut sums = [0.0f32; 4];
        for t in 0..thread_num {
            let start = t * topk_size;
            let local_partial: f32 = values[start..start + topk_size]
                .iter()
                .map(|&v| (v - global_max).exp())
                .sum();
            let first_val = values[start];
            sums[t] = (global_max - first_val).exp() * (local_partial + share_per_thread);
        }

        let mut out_vals = vec![0.0f32; topk_size];
        let mut out_idx = vec![0usize; topk_size];
        let mut out_token = 0usize;

        unsafe {
            topk_softmax(
                values.as_ptr(),
                indices.as_ptr(),
                sums.as_ptr(),
                out_vals.as_mut_ptr(),
                out_idx.as_mut_ptr(),
                &mut out_token,
                thread_num,
                topk_size,
            );
        }

        let mut paired: Vec<(f32, usize)> = values
            .iter()
            .copied()
            .zip(indices.iter().copied())
            .collect();
        paired.sort_by(|a, b| b.0.total_cmp(&a.0));
        let denom: f32 = values.iter().map(|&v| (v - global_max).exp()).sum();

        for i in 0..topk_size {
            let expected_prob = (paired[i].0 - global_max).exp() / denom;
            assert_ulps_eq!(out_vals[i], expected_prob, max_ulps = 4);
            assert_eq!(out_idx[i], paired[i].1);
        }
        assert_eq!(out_token, paired[0].1);
    }

}
