use super::activation::exp512;
use crate::kernel::common::heap::FixedMinHeap;
use std::arch::x86_64::*;
use std::f16;
use std::ptr;

#[target_feature(enable = "avx512f", enable = "avx512fp16")]
unsafe fn truncated_topk_softmax(
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
    let total_candidates = thread_num * topk_size;
    let mut heap = FixedMinHeap::new(output_values_ptr, output_indices_ptr, topk_size);
    for i in 0..total_candidates {
        let value = *input_values_ptr.add(i);
        let index = *input_indices_ptr.add(i);
        heap.push(value, index);
    }

    heap.sort_desc();

    let len = heap.len();
    debug_assert_eq!(len % 32, 0);
    let max_val = *output_values_ptr;

    let max_broadcast = _mm512_set1_ph(max_val);
    for offset in (0..len).step_by(32) {
        let chunk = _mm512_loadu_ph(output_values_ptr.add(offset) as *const u8);
        let shifted = _mm512_sub_ph(chunk, max_broadcast);
        let exp_chunk = exp512(shifted);
        _mm512_storeu_ph(output_values_ptr.add(offset) as *mut u8, exp_chunk);
    }

    let mut total_sum = f16::from_f32(0.0);
    for k in 0..len {
        total_sum = f16::from_f32(total_sum.to_f32() + (*output_values_ptr.add(k)).to_f32());
    }

    let inv_vec = _mm512_set1_ph(f16::from_f32(1.0) / total_sum);
    for offset in (0..len).step_by(32) {
        let chunk = _mm512_loadu_ph(output_values_ptr.add(offset) as *const u8);
        let normalized = _mm512_mul_ph(chunk, inv_vec);
        _mm512_storeu_ph(output_values_ptr.add(offset) as *mut u8, normalized);
    }

    ptr::write(output_token_ptr, *output_indices_ptr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_topk_softmax_matches_simd_version() {
        if !is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }
        let topk_size = 32usize;
        let thread_num = 4usize;
        let values_f32: [f32; 128] = [
            5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4,
            3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7,
            1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
            -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4,
            -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -2.6, -2.7, -2.8,
            -2.9, -3.0, -3.1, -3.2, -3.3, -3.4, -3.5, -3.6, -3.7, -3.8, -3.9, -4.0, -4.1, -4.2,
            -4.3, -4.4, -4.5, -4.6, -4.7, -4.8, -4.9, -5.0, -5.1, -5.2, -5.3, -5.4, -5.5, -5.6,
            -5.7, -5.8, -5.9, -6.0, -6.1, -6.2, -6.3, -6.4, -6.5, -6.6, -6.7,
        ];
        let values: Vec<f16> = values_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let indices: Vec<usize> = (0..topk_size * thread_num).collect();

        let mut out_vals = vec![f16::from_f32(0.0); topk_size];
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

        let mut paired: Vec<(f16, usize)> = values
            .iter()
            .copied()
            .zip(indices.iter().copied())
            .collect();
        paired.sort_by(|a, b| b.0.total_cmp(&a.0));

        let topk = &paired[..topk_size];
        let max_val = topk[0].0;
        let denom: f32 = topk
            .iter()
            .map(|(v, _)| (v.to_f32() - max_val.to_f32()).exp())
            .sum();

        for i in 0..topk_size {
            let expected_prob = (topk[i].0.to_f32() - max_val.to_f32()).exp() / denom;
            assert_ulps_eq!(out_vals[i].to_f32(), expected_prob, max_ulps = 4);
            assert_eq!(out_idx[i], topk[i].1);
        }
        assert_eq!(out_token, topk[0].1);
    }
}
