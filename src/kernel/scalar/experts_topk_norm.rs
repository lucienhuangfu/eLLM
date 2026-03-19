use std::ops::{AddAssign, Div};

pub fn experts_topk_norm<T>(
    ptr1: *const T,
    topk_values_ptr: *mut T,
    experts_indicator: *mut bool,
    indice_ptr: *mut bool,
    value_ptr: *mut T,
    topk_indices_ptr: *mut usize,
    token_index: usize,
    batch_size: usize,
    input_length: usize,
    output_length: usize,
) where
    T: Copy
        + PartialOrd
        + PartialEq
        + Default
        + AddAssign
        + Div<Output = T>,
{
    unsafe {
        let input_slice = std::slice::from_raw_parts(ptr1, input_length);
        let mut indexed_values: Vec<(usize, T)> = input_slice
            .iter()
            .enumerate()
            .map(|(expert_idx, &value)| (expert_idx, value))
            .collect();

        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let topk_items = &indexed_values[..output_length.min(input_length)];

        let mut norm_sum = T::default();
        for &(expert_idx, value) in topk_items {
            norm_sum += value;
            *experts_indicator.add(expert_idx) = true;
        }
        let norm_sum = if norm_sum == T::default() {
            T::default()
        } else {
            norm_sum
        };

        for (k, &(expert_idx, value)) in topk_items.iter().enumerate() {
            let prob = if norm_sum == T::default() {
                value
            } else {
                value / norm_sum
            };
            *topk_values_ptr.add(k) = prob;
            *topk_indices_ptr.add(k) = expert_idx;
            let offset = expert_idx * batch_size + token_index;
            *indice_ptr.add(offset) = true;
            *value_ptr.add(offset) = prob;
        }
    }
}
