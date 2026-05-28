use crate::kernel::common::heap::FixedMinHeap;
use crate::kernel::x86_64::f32_256::activation::exp256;
use core::arch::x86_64::*;

pub fn truncated_topk_softmax(
    // [thread_num, top_k]
    input_values_ptr: *const f32,
    // [thread_num, top_k]
    input_indices_ptr: *const usize,
    temperature: f32,
    // [thread_num]
    // _sums_ptr: *const f32,
    // [top_k]
    output_values_ptr: *mut f32,
    // [top_k]
    output_indices_ptr: *mut usize,
    // [1]
    // output_token_ptr: *mut usize,
    thread_num: usize,
    top_k: usize,
    top_k_simd: usize,
) {
    let total_candidates = thread_num * top_k;
    let mut heap = FixedMinHeap::new(output_values_ptr, output_indices_ptr, top_k);
    for i in 0..total_candidates {
        let value = unsafe { *input_values_ptr.add(i) };
        let index = unsafe { *input_indices_ptr.add(i) };
        heap.push(value, index);
    }

    heap.sort_desc();

    let len = heap.len();
    if len == 0 {
        return;
    }
    debug_assert!(top_k_simd >= len);
    debug_assert_eq!(top_k_simd % 8, 0);
    let max_val = unsafe { *output_values_ptr };

    // let mut sum_vec = _mm256_setzero_ps();
    let max_broadcast = unsafe { _mm256_set1_ps(max_val) };
    let inv_temperature = unsafe { _mm256_set1_ps(1.0f32 / temperature) };

    for i in len..top_k_simd {
        unsafe {
            *output_values_ptr.add(i) = f32::NEG_INFINITY;
            *output_indices_ptr.add(i) = 0usize;
        }
    }

    for offset in (0..top_k_simd).step_by(8) {
        unsafe {
            let chunk = _mm256_loadu_ps(output_values_ptr.add(offset));
            let shifted = _mm256_mul_ps(_mm256_sub_ps(chunk, max_broadcast), inv_temperature);
            let exp_chunk = exp256(shifted);
            _mm256_storeu_ps(output_values_ptr.add(offset), exp_chunk);
        }
    }

    let mut total_sum = 0.0f32;
    for k in 0..top_k_simd {
        total_sum += unsafe { *output_values_ptr.add(k) };
    }

    let inv_vec = unsafe { _mm256_set1_ps(1.0f32 / total_sum) };
    for offset in (0..top_k_simd).step_by(8) {
        unsafe {
            let chunk = _mm256_loadu_ps(output_values_ptr.add(offset));
            let normalized = _mm256_mul_ps(chunk, inv_vec);
            _mm256_storeu_ps(output_values_ptr.add(offset), normalized);
        }
    }

    // unsafe { ptr::write(output_token_ptr, *output_indices_ptr) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_topk_softmax_matches_simd_version() {
        let top_k = 8usize;
        let thread_num = 4usize;
        let values: [f32; 32] = [
            5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4,
            3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9,
        ];
        let indices: Vec<usize> = (0..top_k * thread_num).collect();

        let mut out_vals = vec![0.0f32; top_k];
        let mut out_idx = vec![0usize; top_k];
        let mut out_token = 0usize;
        let sums = vec![0.0f32; thread_num];

        truncated_topk_softmax(
            values.as_ptr(),
            indices.as_ptr(),
            1.0f32,
            // sums.as_ptr(),
            out_vals.as_mut_ptr(),
            out_idx.as_mut_ptr(),
            // &mut out_token,
            thread_num,
            top_k,
            top_k,
        );

        let mut paired: Vec<(f32, usize)> = values
            .iter()
            .copied()
            .zip(indices.iter().copied())
            .collect();
        paired.sort_by(|a, b| b.0.total_cmp(&a.0));

        let topk = &paired[..top_k];
        let max_val = topk[0].0;
        let denom: f32 = topk.iter().map(|(v, _)| (v - max_val).exp()).sum();

        for i in 0..top_k {
            let expected_prob = (topk[i].0 - max_val).exp() / denom;
            assert_ulps_eq!(out_vals[i], expected_prob, max_ulps = 4);
            assert_eq!(out_idx[i], topk[i].1);
        }
        assert_eq!(out_token, topk[0].1);
    }
}
