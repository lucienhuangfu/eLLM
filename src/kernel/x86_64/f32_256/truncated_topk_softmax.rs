use crate::kernel::common::heap::FixedMinHeap;
use crate::kernel::x86_64::f32_256::activation::exp256;
use core::arch::x86_64::*;
use std::ptr;

unsafe fn truncated_topk_softmax(
    // [thread_num, topk_size]
    input_values_ptr: *const f32,
    // [thread_num, topk_size]
    input_indices_ptr: *const usize,

    // [thread_num]
    // sums_ptr: *const f32,
    // [topk_size]
    output_values_ptr: *mut f32,
    // [topk_size]
    output_indices_ptr: *mut usize,
    // [1]
    output_token_ptr: *mut usize,
    thread_num: usize,
    topk_size: usize,
) {
    let total_candidates = thread_num * topk_size;
    let mut heap = FixedMinHeap::new(output_values_ptr, output_indices_ptr, topk_size);
    for i in 0..total_candidates {
        let value = *input_values_ptr.add(i);
        let index = *input_indices_ptr.add(i);
        heap.push(value, index);
    }

    heap.sort_desc();

    let len = heap.len();
    debug_assert_eq!(len % 8, 0);
    let max_val = *output_values_ptr;

    let mut sum_vec = _mm256_setzero_ps();
    let max_broadcast = _mm256_set1_ps(max_val);
    let mut i = 0;
    while i < len {
        let chunk = _mm256_loadu_ps(output_values_ptr.add(i));
        let shifted = _mm256_sub_ps(chunk, max_broadcast);
        let exp_chunk = exp256(shifted);
        _mm256_storeu_ps(output_values_ptr.add(i), exp_chunk);
        sum_vec = _mm256_add_ps(sum_vec, exp_chunk);
        i += 8;
    }

    let mut sum_buf = [0f32; 8];
    _mm256_storeu_ps(sum_buf.as_mut_ptr(), sum_vec);
    let total_sum = sum_buf.iter().sum::<f32>();
    let inv_vec = _mm256_set1_ps(1.0f32 / total_sum);
    let mut j = 0;
    while j < len {
        let chunk = _mm256_loadu_ps(output_values_ptr.add(j));
        let normalized = _mm256_mul_ps(chunk, inv_vec);
        _mm256_storeu_ps(output_values_ptr.add(j), normalized);
        j += 8;
    }

    ptr::write(output_token_ptr, *output_indices_ptr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_topk_softmax_matches_simd_version() {
        let topk_size = 8usize;
        let thread_num = 4usize;
        let values: [f32; 32] = [
            5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4,
            3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9,
        ];
        let indices: Vec<usize> = (0..topk_size * thread_num).collect();

        let mut out_vals = vec![0.0f32; topk_size];
        let mut out_idx = vec![0usize; topk_size];
        let mut out_token = 0usize;

        unsafe {
            truncated_topk_softmax(
                values.as_ptr(),
                indices.as_ptr(),
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

        let topk = &paired[..topk_size];
        let max_val = topk[0].0;
        let denom: f32 = topk.iter().map(|(v, _)| (v - max_val).exp()).sum();

        for i in 0..topk_size {
            let expected_prob = (topk[i].0 - max_val).exp() / denom;
            assert_ulps_eq!(out_vals[i], expected_prob, max_ulps = 4);
            assert_eq!(out_idx[i], topk[i].1);
        }
        assert_eq!(out_token, topk[0].1);
    }
}
