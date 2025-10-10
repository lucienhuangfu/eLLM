use crate::kernel::generic::exp::Exp;
use std::ops::{AddAssign, Div, Sub};
use std::ptr;

#[inline]
fn heapify_down<T: PartialOrd + Copy>(
    input_ptr: *const T,
    indices_ptr: *mut usize,
    heap_size: usize,
    start: usize,
) {
    unsafe {
        let mut parent = start;
        loop {
            let left = 2 * parent + 1;
            let right = 2 * parent + 2;
            let mut smallest = parent;

            if left < heap_size {
                let left_idx = *indices_ptr.add(left);
                let smallest_idx = *indices_ptr.add(smallest);
                if *input_ptr.add(left_idx) < *input_ptr.add(smallest_idx) {
                    smallest = left;
                }
            }

            if right < heap_size {
                let right_idx = *indices_ptr.add(right);
                let smallest_idx = *indices_ptr.add(smallest);
                if *input_ptr.add(right_idx) < *input_ptr.add(smallest_idx) {
                    smallest = right;
                }
            }

            if smallest != parent {
                // Swap
                let temp = *indices_ptr.add(parent);
                *indices_ptr.add(parent) = *indices_ptr.add(smallest);
                *indices_ptr.add(smallest) = temp;
                parent = smallest;
            } else {
                break;
            }
        }
    }
}

#[inline]
fn build_min_heap<T: PartialOrd + Copy>(
    input_ptr: *const T,
    indices_ptr: *mut usize,
    heap_size: usize,
) {
    unsafe {
        // Initialize with first heap_size indices
        for i in 0..heap_size {
            *indices_ptr.add(i) = i;
        }

        // Heapify from bottom up
        if heap_size > 0 {
            for i in (0..heap_size).rev() {
                heapify_down(input_ptr, indices_ptr, heap_size, i);
            }
        }
    }
}

#[inline]
fn select_topk<T: PartialOrd + Copy>(
    input_ptr: *const T,
    indices_ptr: *mut usize,
    length: usize,
    topk_size: usize,
) {
    unsafe {
        // Build initial heap
        let heap_size = topk_size.min(length);
        build_min_heap(input_ptr, indices_ptr, heap_size);

        // Process remaining elements
        for i in topk_size..length {
            let current_val = *input_ptr.add(i);
            let root_idx = *indices_ptr.add(0);
            let root_val = *input_ptr.add(root_idx);

            if current_val > root_val {
                *indices_ptr.add(0) = i;
                heapify_down(input_ptr, indices_ptr, topk_size, 0);
            }
        }
    }
}

// Optimized version using partial selection for small k
#[inline]
fn select_topk_optimized<T: PartialOrd + Copy>(
    input_ptr: *const T,
    indices_ptr: *mut usize,
    length: usize,
    topk_size: usize,
) {
    unsafe {
        // For small k, use partial selection instead of heap
        if topk_size <= 8 && topk_size < length / 4 {
            // Initialize with first topk_size elements
            for i in 0..topk_size {
                *indices_ptr.add(i) = i;
            }

            // For each remaining element, find insertion point
            for i in topk_size..length {
                let current_val = *input_ptr.add(i);

                // Find minimum in current topk
                let mut min_pos = 0;
                let mut min_val = *input_ptr.add(*indices_ptr.add(0));

                for j in 1..topk_size {
                    let val = *input_ptr.add(*indices_ptr.add(j));
                    if val < min_val {
                        min_val = val;
                        min_pos = j;
                    }
                }

                // Replace if current is larger
                if current_val > min_val {
                    *indices_ptr.add(min_pos) = i;
                }
            }
        } else {
            // Use heap for larger k
            select_topk(input_ptr, indices_ptr, length, topk_size);
        }
    }
}

#[inline]
fn sort_topk_descending<T: PartialOrd + Copy>(
    input_ptr: *const T,
    indices_ptr: *mut usize,
    topk_size: usize,
) {
    unsafe {
        // Convert min-heap to sorted array (largest to smallest)
        for i in (1..topk_size).rev() {
            // Move current root to end
            let temp = *indices_ptr.add(0);
            *indices_ptr.add(0) = *indices_ptr.add(i);
            *indices_ptr.add(i) = temp;

            // Heapify reduced heap (but reverse comparison for descending order)
            let mut parent = 0;
            loop {
                let left = 2 * parent + 1;
                let right = 2 * parent + 2;
                let mut largest = parent;

                if left < i {
                    let left_idx = *indices_ptr.add(left);
                    let largest_idx = *indices_ptr.add(largest);
                    if *input_ptr.add(left_idx) > *input_ptr.add(largest_idx) {
                        largest = left;
                    }
                }

                if right < i {
                    let right_idx = *indices_ptr.add(right);
                    let largest_idx = *indices_ptr.add(largest);
                    if *input_ptr.add(right_idx) > *input_ptr.add(largest_idx) {
                        largest = right;
                    }
                }

                if largest != parent {
                    let temp = *indices_ptr.add(parent);
                    *indices_ptr.add(parent) = *indices_ptr.add(largest);
                    *indices_ptr.add(largest) = temp;
                    parent = largest;
                } else {
                    break;
                }
            }
        }
    }
}

#[inline]
fn calculate_softmax_normalized<
    T: Exp + Default + AddAssign + Copy + Sub<Output = T> + Div<Output = T>,
>(
    input_ptr: *const T,
    indices_ptr: *const usize,
    output_ptr: *mut T,
    topk_size: usize,
) {
    unsafe {
        if topk_size == 0 {
            return;
        }

        // Max value is first element (already sorted)
        let max_idx = *indices_ptr.add(0);
        let max_val = *input_ptr.add(max_idx);

        // Calculate exp values and sum
        let mut sum = T::default();
        for i in 0..topk_size {
            let idx = *indices_ptr.add(i);
            let exp_val = (*input_ptr.add(idx) - max_val).exp();
            *output_ptr.add(i) = exp_val;
            sum += exp_val;
        }

        let mut norm_sum = T::default();
        // Normalize
        for i in 0..topk_size {
            let exp_val = *output_ptr.add(i);
            *output_ptr.add(i) = exp_val / sum;
            norm_sum += *output_ptr.add(i);
        }

        // 需要对topk的结果进行归一化处理，使得所有值的和为1
        for i in 0..topk_size {
            let val = *output_ptr.add(i);
            *output_ptr.add(i) = val / norm_sum;
        }
    }
}

pub fn experts_topk_softmax_norm<
    T: Exp + Default + AddAssign + PartialOrd + Copy + Sub<Output = T> + Div<Output = T>,
>(
    input_ptr: *const T,
    output_indices_ptr: *mut usize,
    output_value_ptr: *mut T,
    length: usize,
    topk_size: usize,
) {
    if topk_size == 0 || length == 0 {
        return;
    }

    // Step 1: Select top-k largest elements (optimized for small k)
    select_topk_optimized(input_ptr, output_indices_ptr, length, topk_size);

    // Step 2: Sort using heap sort (more efficient than insertion sort)
    sort_topk_descending(input_ptr, output_indices_ptr, topk_size);

    // Step 3: Calculate softmax with double normalization
    calculate_softmax_normalized(input_ptr, output_indices_ptr, output_value_ptr, topk_size);
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
