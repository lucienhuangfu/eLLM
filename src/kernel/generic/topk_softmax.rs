use crate::kernel::generic::exp::Exp;
use std::ops::{AddAssign, Div, Sub};
use std::ptr;

pub fn topk_softmax<
    T: Exp + Default + AddAssign + PartialOrd + Copy + Sub<Output = T> + Div<Output = T>,
>(
    // [thread_num, topk_size]
    input_indices_ptr: *const usize,
    // [thread_num, topk_size]
    input_values_ptr: *const T,
    // [thread_num]
    sums_ptr: *const T,
    // [topk_size]
    output_indices_ptr: *mut usize,
    // [topk_size]
    output_values_ptr: *mut T,
    thread_num: usize,
    topk_size: usize,
) {
    unsafe {
        // Track current position for each thread
        let mut thread_positions = vec![0usize; thread_num];
        let mut merged_pairs: Vec<(T, usize)> = Vec::with_capacity(topk_size);

        // Find global max for numerical stability
        let mut max_val = T::default();
        let mut first = true;
        for thread_idx in 0..thread_num {
            if thread_positions[thread_idx] < topk_size {
                let val =
                    *input_values_ptr.add(thread_idx * topk_size + thread_positions[thread_idx]);
                if first {
                    max_val = val;
                    first = false;
                } else if val > max_val {
                    max_val = val;
                }
            }
        }

        // Merge sorted topk lists to get global topk
        for _ in 0..topk_size {
            let mut best_thread = None;
            let mut best_val = T::default();

            // Find thread with highest current value
            for thread_idx in 0..thread_num {
                if thread_positions[thread_idx] < topk_size {
                    let val = *input_values_ptr
                        .add(thread_idx * topk_size + thread_positions[thread_idx]);
                    if best_thread.is_none() || val > best_val {
                        best_thread = Some(thread_idx);
                        best_val = val;
                    }
                }
            }

            if let Some(thread_idx) = best_thread {
                let pos = thread_positions[thread_idx];
                let idx = *input_indices_ptr.add(thread_idx * topk_size + pos);
                merged_pairs.push((best_val, idx));
                thread_positions[thread_idx] += 1;
            } else {
                break;
            }
        }

        // Apply softmax to merged results
        let mut sum = T::default();
        for (val, _) in &mut merged_pairs {
            *val = (*val - max_val).exp();
            sum += *val;
        }

        // Normalize and write output
        for i in 0..merged_pairs.len() {
            let normalized_val = merged_pairs[i].0 / sum;
            ptr::write(output_values_ptr.add(i), normalized_val);
            ptr::write(output_indices_ptr.add(i), merged_pairs[i].1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

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
