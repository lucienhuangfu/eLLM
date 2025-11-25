use std::arch::x86_64::*;
use std::f16;
use std::ptr;

use super::activation::exp512;
use crate::kernel::common::heap::FixedMinHeap;

// #[target_feature(enable = "avx512f", enable = "avx512fp16")]
pub fn truncated_topk_softmax(
    // [thread_num, topk_size]
    input_values_ptr: *const f16,
    // [thread_num, topk_size]
    input_indices_ptr: *const usize,
    // [topk_size]
    output_values_ptr: *mut f16,
    // [topk_size]
    output_indices_ptr: *mut usize,
    // [1]
    output_token_ptr: *mut usize,
    thread_num: usize,
    topk_size: usize,
) {
    unsafe {
        let total_candidates = thread_num * topk_size;
        let mut heap = FixedMinHeap::new(output_values_ptr, output_indices_ptr, topk_size);
        for i in 0..total_candidates {
            let value = *input_values_ptr.add(i);
            let index = *input_indices_ptr.add(i);
            heap.push(value, index);
        }

        heap.sort_desc();

        let len = heap.len();
        let max_val = *output_values_ptr;

        let mut buffer = [f16::NEG_INFINITY; 32];
        ptr::copy_nonoverlapping(output_values_ptr, buffer.as_mut_ptr(), len);

        let max_broadcast = _mm512_set1_ph(max_val);
        let chunk = _mm512_loadu_ph(buffer.as_ptr());
        let shifted = _mm512_sub_ph(chunk, max_broadcast);
        let exp_chunk = exp512(shifted);

        let total_sum = _mm512_reduce_add_ph(exp_chunk);

        let inv_vec = _mm512_set1_ph(total_sum.recip());
        let normalized = _mm512_mul_ph(exp_chunk, inv_vec);

        _mm512_storeu_ph(buffer.as_mut_ptr(), normalized);
        ptr::copy_nonoverlapping(buffer.as_ptr(), output_values_ptr, len);

        ptr::write(output_token_ptr, *output_indices_ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_topk_softmax_matches_simd_version() {
        if !is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }
        let topk_size = 8usize;
        let thread_num = 4usize;
        let values: [f16; 128] = [
            5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4,
            3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7,
            1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
            -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4,
            -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -2.6, -2.7, -2.8,
            -2.9, -3.0, -3.1, -3.2, -3.3, -3.4, -3.5, -3.6, -3.7, -3.8, -3.9, -4.0, -4.1, -4.2,
            -4.3, -4.4, -4.5, -4.6, -4.7, -4.8, -4.9, -5.0, -5.1, -5.2, -5.3, -5.4, -5.5, -5.6,
            -5.7, -5.8, -5.9, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7, -6.8, -6.9, -7.0,
            -7.1, -7.2, -7.3, -7.4, -7.5, -7.6, -7.7,
        ];
        // let values: Vec<f16> = values_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let indices: Vec<usize> = (0..topk_size * thread_num).collect();

        let mut out_vals = vec![0.0f16; topk_size];
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

        let total_candidates = topk_size * thread_num;
        let mut paired: Vec<(f16, usize)> = values
            .iter()
            .take(total_candidates)
            .copied()
            .zip(indices.iter().copied())
            .collect();
        paired.sort_by(|a, b| b.0.total_cmp(&a.0));

        let topk = &paired[..topk_size];
        let max_val = topk[0].0;
        let denom: f32 = topk
            .iter()
            .map(|(v, _)| ((*v as f32) - (max_val as f32)).exp())
            .sum();

        for i in 0..topk_size {
            let expected_prob = ((topk[i].0 as f32) - (max_val as f32)).exp() / denom;
            assert_abs_diff_eq!((out_vals[i] as f32), expected_prob, epsilon = 1e-3);
            assert_eq!(out_idx[i], topk[i].1);
        }
        assert_eq!(out_token, topk[0].1);
    }
}
