use std::ptr;
use std::arch::x86_64::*;
use crate::kernel::common::heap::FixedMinHeap;

use crate::kernel::x86_64::f32_256::bitonic_sort::bitonic_sort_f32x8_desc;

pub fn topk_softmax(
    // [thread_num, topk_size]
    input_values_ptr: *const f32,
    // [thread_num, topk_size]
    input_indices_ptr: *const usize,

    // [thread_num]
    sums_ptr: *const f32,
    // max_positions_ptr: *mut usize,
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
            thread_num *  topk_size,
            topk_size,
        );

        // Get max value directly from first element (merge sort results are ordered)
        let max_val = *output_values_ptr.add(0);
        // Calculate adjusted total sum (subtract max for numerical stability)
        let mut total_sum = f32::zero();
        for i in 0..thread_num {
            total_sum += (*sums_ptr.add(i))*(*input_values_ptr.add(i*topk_size) - max_val).exp();
        }

        total_sum -= (max_val * topk_size * thread_num);

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
    in_indices: *const i32,
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
        let idx_vec = _mm256_loadu_si256(in_indices.add(chunk_start) as *const __m256i);
        let (sorted_vals, sorted_idx) = bitonic_sort_f32x8_desc(values, idx_vec);

        let mut chunk_vals = [0.0f32; 8];
        let mut chunk_idx = [0i32; 8];
        _mm256_storeu_ps(chunk_vals.as_mut_ptr(), sorted_vals);
        _mm256_storeu_si256(chunk_idx.as_mut_ptr() as *mut __m256i, sorted_idx);

        let chunk_take = topk.min(8);
        for lane in 0..chunk_take {
            heap.push(chunk_vals[lane], chunk_idx[lane] as usize);
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
        let indices = [10, 21, 5, 42, 7, 18, 33, 2, 44, 13, 8, 29, 1, 55, 3, 60];
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

        let mut expected: Vec<(f32, i32)> =
            data.iter().copied().zip(indices.iter().copied()).collect();
        expected.sort_by(|a, b| b.0.total_cmp(&a.0));

        for i in 0..topk {
            assert_ulps_eq!(out_vals[i], expected[i].0);
            assert_eq!(out_idx[i], expected[i].1 as usize);
        }
    }

    /*
    #[test]
    fn test_scale_softmax() {
        let v1: Vec<f32> = (1..19).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; v1.len()];
        scale_softmax(v1.as_ptr(), output.as_mut_ptr(), v1.len(), 0.65);
        let result: [f32; 18] = [
            7.5933926382276695e-06,
            1.4545462363457773e-05,
            2.7862415663548745e-05,
            5.337157563189976e-05,
            0.00010223548451904207,
            0.00019583618268370628,
            0.0003751322510652244,
            0.0007185811409726739,
            0.0013764717150479555,
            0.0026366880629211664,
            0.005050681531429291,
            0.009674787521362305,
            0.01853245310485363,
            0.035499654710292816,
            0.06800107657909393,
            0.13025879859924316,
            0.24951595067977905,
            0.4779582619667053,
        ];
        assert_ulps_eq!(output[..], result, max_ulps = 4);
    }*/
}
