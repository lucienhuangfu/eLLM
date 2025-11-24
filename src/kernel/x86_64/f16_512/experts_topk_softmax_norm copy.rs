use crate::kernel::common::heap::FixedMinHeap;
use crate::kernel::generic::exp::Exp;
use crate::kernel::x86_64::f16_512::activation::exp512;
use crate::kernel::x86_64::f16_512::bitonic_sort::bitonic_sort_f16_desc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::f16;

const LANES: usize = 32;



#[inline(always)]
unsafe fn sum_f16_vec(vec: __m512h, count: usize) -> f32 {
    debug_assert!(count <= LANES);
    let mut tmp = [0.0f16; LANES];
    _mm512_storeu_ph(tmp.as_mut_ptr() as *mut _, vec);
    tmp.iter()
        .take(count)
        .fold(0.0f32, |acc, &v| acc + (v as f32))
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
            let local_idx = chunk_idx[i] as usize;
            let global_idx = chunk_start + local_idx;
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
        
        let mut sum = 0.0;
        for i in 0..num_topk {
            sum += *topk_values_ptr.add(i);
        }
        let scale = sum.recip();
        for i in 0..num_topk {
            *topk_values_ptr.add(i) *= scale;
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
    // use half::f16;
    use std::arch::is_x86_feature_detected;

    const DATA: [f16; 32] = [
        0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        12.25, -4.75, 13.5, 14.75, 15.5, -6.0, 16.25, 17.5, 18.75, -7.5, 19.25, 20.5, 21.75, -8.25,
        22.0, 23.5,
    ];

    fn features_available() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512fp16")
    }

    #[test]
    fn test_get_topk() {
        if !features_available() {
            eprintln!("Skipping AVX-512 FP16-dependent test");
            return;
        }
        const TOPK: usize = 8;
        let data: Vec<f16> = DATA.iter().map(|&v| v).collect();
        let mut out_vals = [0.0f16; TOPK];
        let mut out_idx = [0usize; TOPK];

        unsafe {
            get_topk(
                data.as_ptr() as *const _,
                out_vals.as_mut_ptr() as *mut _,
                out_idx.as_mut_ptr(),
                DATA.len(),
                TOPK,
            );
        }

        let raw_vals = out_vals;
        let mut expected: Vec<(f32, usize)> = DATA
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, val)| (val, idx))
            .collect();
        expected.sort_by(|a, b| b.0.total_cmp(&a.0));

        for i in 0..TOPK {
            assert_ulps_eq!(raw_vals[i].to_f32(), expected[i].0, max_ulps = 16);
            assert_eq!(out_idx[i], expected[i].1);
        }
    }

    #[test]
    fn test_softmax_stats_avx512_fp16() {
        if !features_available() {
            eprintln!("Skipping AVX-512 FP16-dependent test");
            return;
        }
        let data: Vec<f16> = DATA.iter().map(|&v| v).collect();
        let expected_max = DATA.iter().copied().fold(f16::NEG_INFINITY, f16::max);
        let expected_denom: f32 = DATA.iter().map(|v| (v - expected_max).exp()).sum();

        let (max_val, denom) =
            unsafe { softmax_stats_avx512_fp16(data.as_ptr() as *const _, DATA.len()) };
        assert_relative_eq!(max_val, expected_max, epsilon = 1e-3);
        assert_relative_eq!(denom, expected_denom, epsilon = 1e-2);
    }

    #[test]
    fn test_softmax_topk_inplace() {
        if !features_available() {
            eprintln!("Skipping AVX-512 FP16-dependent test");
            return;
        }
        const TOPK: usize = 8;
        let mut expected: Vec<(f32, usize)> = DATA
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, val)| (val, idx))
            .collect();
        expected.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut topk_vals = [f16::ZERO; TOPK];
        for i in 0..TOPK {
            topk_vals[i] = f16::from_f32(expected[i].0);
        }

        let max_val = expected[0].0;
        let denom: f32 = DATA.iter().map(|v| (v - max_val).exp()).sum();

        unsafe {
            softmax_topk_inplace(topk_vals.as_mut_ptr() as *mut _, TOPK, max_val, denom);
        }

        for i in 0..TOPK {
            let expected_prob = (expected[i].0 - max_val).exp() / denom;
            assert_relative_eq!(topk_vals[i].to_f32(), expected_prob, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_experts_topk_softmax_norm() {
        if !features_available() {
            eprintln!("Skipping AVX-512 FP16-dependent test");
            return;
        }
        const NUM_EXPERTS: usize = 32;
        const NUM_TOPK: usize = 8;
        const NUM_TOKEN: usize = 3;
        const INDEX_TOKEN: usize = 1;

        let data: Vec<f16> = DATA.iter().map(|&v| f16::from_f32(v)).collect();
        let mut topk_vals = [f16::ZERO; NUM_TOPK];
        let mut topk_idx = [0usize; NUM_TOPK];
        let mut expert_flags = [false; NUM_EXPERTS];
        let mut indices = [false; NUM_EXPERTS * NUM_TOKEN];
        let mut values = vec![f16::ZERO; NUM_EXPERTS * NUM_TOKEN];

        experts_topk_softmax_norm(
            data.as_ptr() as *const _,
            topk_vals.as_mut_ptr() as *mut _,
            topk_idx.as_mut_ptr(),
            expert_flags.as_mut_ptr(),
            indices.as_mut_ptr(),
            values.as_mut_ptr() as *mut _,
            INDEX_TOKEN,
            NUM_TOKEN,
            NUM_EXPERTS,
            NUM_TOPK,
        );

        let mut expected: Vec<(usize, f32)> = DATA.iter().copied().enumerate().collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1));

        let max_val = DATA.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = DATA.iter().map(|v| (v - max_val).exp()).sum();

        let mut is_topk = [false; NUM_EXPERTS];
        for i in 0..NUM_TOPK {
            let (idx, value) = expected[i];
            let prob = (value - max_val).exp() / denom;
            assert_eq!(topk_idx[i], idx);
            assert_relative_eq!(topk_vals[i].to_f32(), prob, epsilon = 1e-2);
            assert!(expert_flags[idx]);
            let offset = idx * NUM_TOKEN + INDEX_TOKEN;
            assert!(indices[offset]);
            assert_relative_eq!(values[offset].to_f32(), prob, epsilon = 1e-2);
            for token in 0..NUM_TOKEN {
                let off = idx * NUM_TOKEN + token;
                if token != INDEX_TOKEN {
                    assert!(!indices[off]);
                    assert_eq!(values[off].to_bits(), f16::ZERO.to_bits());
                }
            }
            is_topk[idx] = true;
        }

        for expert in 0..NUM_EXPERTS {
            if !is_topk[expert] {
                assert!(!expert_flags[expert]);
                for token in 0..NUM_TOKEN {
                    let off = expert * NUM_TOKEN + token;
                    assert!(!indices[off]);
                    assert_eq!(values[off].to_bits(), f16::ZERO.to_bits());
                }
            }
        }
    }
}
