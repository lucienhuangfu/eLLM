use std::ptr;

pub unsafe fn merge_topk_lists<T: PartialOrd + Copy + Default>(
    input_indices_ptr: *const usize,
    input_values_ptr: *const T,
    max_positions_ptr: *mut usize,
    output_indices_ptr: *mut usize,
    output_values_ptr: *mut T,
    thread_num: usize,
    topk_size: usize,
) -> usize {
    let mut merged_count = 0;

    // Initialize positions to zero using write_bytes for better performance
    ptr::write_bytes(max_positions_ptr, 0, thread_num);

    // Merge sorted topk lists from all threads using multi-way merge
    for _ in 0..topk_size {
        let mut best_value = None;
        let mut best_thread = None;

        // Find thread with maximum current value (merge step)
        for thread_idx in 0..thread_num {
            let current_pos = *max_positions_ptr.add(thread_idx);
            if current_pos < topk_size {
                let val = *input_values_ptr.add(thread_idx * topk_size + current_pos);
                if best_value.is_none() || val > best_value.unwrap() {
                    best_value = Some(val);
                    best_thread = Some(thread_idx);
                }
            }
        }

        if let (Some(thread_idx), Some(value)) = (best_thread, best_value) {
            let pos = *max_positions_ptr.add(thread_idx);
            let idx = *input_indices_ptr.add(thread_idx * topk_size + pos);

            // Write directly to output
            ptr::write(output_values_ptr.add(merged_count), value);
            ptr::write(output_indices_ptr.add(merged_count), idx);
            merged_count += 1;

            // Move to next element in the selected thread's topk list
            let new_pos = pos + 1;
            ptr::write(max_positions_ptr.add(thread_idx), new_pos);
        } else {
            break;
        }
    }
    merged_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f16;

    #[test]
    fn test_merge_topk_lists_f16() {
        unsafe {
            // Test data: 2 threads, each with topk_size=3
            let thread_num = 2;
            let topk_size = 3;

            // Thread 0: values [5.0, 3.0, 1.0], indices [10, 30, 50]
            // Thread 1: values [4.0, 2.0, 0.5], indices [20, 40, 60]
            let input_values = vec![
                f16::from_f32(5.0),
                f16::from_f32(3.0),
                f16::from_f32(1.0), // thread 0
                f16::from_f32(4.0),
                f16::from_f32(2.0),
                f16::from_f32(0.5), // thread 1
            ];
            let input_indices = vec![10, 30, 50, 20, 40, 60];

            let mut max_positions = vec![0usize; thread_num];
            let mut output_values = vec![f16::ZERO; topk_size];
            let mut output_indices = vec![0usize; topk_size];

            let merged_count = merge_topk_lists(
                input_indices.as_ptr(),
                input_values.as_ptr(),
                max_positions.as_mut_ptr(),
                output_indices.as_mut_ptr(),
                output_values.as_mut_ptr(),
                thread_num,
                topk_size,
            );

            // Expected merged result: [5.0, 4.0, 3.0] with indices [10, 20, 30]
            assert_eq!(merged_count, 3);
            assert_eq!(output_values[0].to_f32(), 5.0);
            assert_eq!(output_values[1].to_f32(), 4.0);
            assert_eq!(output_values[2].to_f32(), 3.0);
            assert_eq!(output_indices[0], 10);
            assert_eq!(output_indices[1], 20);
            assert_eq!(output_indices[2], 30);
        }
    }

    #[test]
    fn test_merge_topk_lists_f32() {
        unsafe {
            // Test with f32 values
            let thread_num = 2;
            let topk_size = 2;

            let input_values = vec![
                10.0f32, 5.0f32, // thread 0
                8.0f32, 3.0f32, // thread 1
            ];
            let input_indices = vec![100, 200, 300, 400];

            let mut max_positions = vec![0usize; thread_num];
            let mut output_values = vec![0.0f32; topk_size];
            let mut output_indices = vec![0usize; topk_size];

            let merged_count = merge_topk_lists(
                input_indices.as_ptr(),
                input_values.as_ptr(),
                max_positions.as_mut_ptr(),
                output_indices.as_mut_ptr(),
                output_values.as_mut_ptr(),
                thread_num,
                topk_size,
            );

            // Expected merged result: [10.0, 8.0] with indices [100, 300]
            assert_eq!(merged_count, 2);
            assert_eq!(output_values[0], 10.0);
            assert_eq!(output_values[1], 8.0);
            assert_eq!(output_indices[0], 100);
            assert_eq!(output_indices[1], 300);
        }
    }

    #[test]
    fn test_merge_empty_lists() {
        unsafe {
            let thread_num = 2;
            let topk_size = 0;

            let input_values = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
            let input_indices = vec![10, 20];

            let mut max_positions = vec![0usize; thread_num];
            let mut output_values = vec![f16::ZERO; 1];
            let mut output_indices = vec![0usize; 1];

            let merged_count = merge_topk_lists(
                input_indices.as_ptr(),
                input_values.as_ptr(),
                max_positions.as_mut_ptr(),
                output_indices.as_mut_ptr(),
                output_values.as_mut_ptr(),
                thread_num,
                topk_size,
            );

            assert_eq!(merged_count, 0);
        }
    }
}
