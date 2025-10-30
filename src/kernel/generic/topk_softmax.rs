use crate::kernel::generic::exp::Exp;
use crate::kernel::generic::merge_topk::merge_topk_lists;
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
    max_positions_ptr: *mut usize,
    // [topk_size]
    output_indices_ptr: *mut usize,
    // [topk_size]
    output_values_ptr: *mut T,
    thread_num: usize,
    topk_size: usize,
) {
    unsafe {
        // Use the generic merge sort function
        let merged_count = merge_topk_lists(
            input_indices_ptr,
            input_values_ptr,
            max_positions_ptr,
            output_indices_ptr,
            output_values_ptr,
            thread_num,
            topk_size,
        );

        // Get max value directly from first element (merge sort results are ordered)
        let max_val = *output_values_ptr.add(0);
        // Calculate adjusted total sum (subtract max for numerical stability)
        let mut total_sum = T::default();
        for i in 0..thread_num {
            total_sum += *sums_ptr.add(i);
        }

        total_sum -= (max_val * topk_size * thread_num);

        // Normalize using the adjusted total sum
        for i in 0..topk_size {
            let val = *output_values_ptr.add(i);
            let exp_val = (val - max_val).exp();
            let normalized_val = exp_val / total_sum;
            ptr::write(output_values_ptr.add(i), normalized_val);
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
