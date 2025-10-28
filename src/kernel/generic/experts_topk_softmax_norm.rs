use crate::kernel::generic::exp::Exp;
use std::ops::{AddAssign, Div, Sub};
use std::ptr;

pub fn experts_topk_softmax_norm<
    T: Exp + Default + AddAssign + PartialOrd + Copy + Sub<Output = T> + Div<Output = T>,
>(
    input_ptr: *const T,
    // [num_experts]
    experts_indicator_ptr: *mut bool,
    // [num_experts, batch_size]
    indices_ptr: *mut bool,
    value_ptr: *mut T,
    index_token: usize,
    num_token: usize,
    num_experts: usize,
    num_topk: usize,
) {
    unsafe {
        // Read input values
        let input_slice = std::slice::from_raw_parts(input_ptr, num_experts);

        // Find top-k indices and values
        let mut indexed_values: Vec<(usize, T)> = input_slice
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Sort by value in descending order
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-k
        let topk_items = &indexed_values[..num_topk.min(num_experts)];

        // Calculate sum of all input values for softmax normalization
        let mut sum = T::default();
        for &value in input_slice {
            sum += value.exp();
        }

        // Set experts_indicator for top-k experts
        for &(expert_idx, _) in topk_items {
            *experts_indicator_ptr.add(expert_idx) = true;
        }

        // For each top-k expert, compute softmax and set outputs
        for (k, &(expert_idx, value)) in topk_items.iter().enumerate() {
            // Compute softmax: exp(x) / sum
            let softmax_value = value.exp() / sum;

            *experts_indicator_ptr.add(expert_idx) = true;
            // Set indices_ptr at [expert_idx * num_token + index_token]
            *indices_ptr.add(expert_idx * num_token + index_token) = true;
            // Set value_ptr
            *value_ptr.add(k) = softmax_value;
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
