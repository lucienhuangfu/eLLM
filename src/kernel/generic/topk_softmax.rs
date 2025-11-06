use crate::kernel::common::heap::FixedMinHeap;
use crate::kernel::generic::exp::Exp;
use std::ops::{AddAssign, Div, Mul, Sub};
use std::ptr;

pub fn topk_softmax<
    T: Exp
        + Default
        + AddAssign
        + PartialOrd
        + Copy
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>,
>(
    // [thread_num, topk_size]
    input_values_ptr: *const T,
    // [thread_num, topk_size]
    input_indices_ptr: *const usize,

    // [thread_num]
    sums_ptr: *const T,
    // [topk_size]
    output_values_ptr: *mut T,
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

        let max_val = *output_values_ptr.add(0);
        let mut total_sum = T::default();
        for i in 0..thread_num {
            let base = *input_values_ptr.add(i * topk_size);
            let adjusted = (base - max_val).exp();
            total_sum += (*sums_ptr.add(i)) * adjusted;
        }

        let mut correction = T::default();
        for _ in 0..total_candidates {
            correction += max_val;
        }
        total_sum = total_sum - correction;

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
