// use crate::kernel::generic::exp::Exp;
#[cfg(target_arch = "x86_64")]
use crate::kernel::x86_64::f32_256::bitonic_sort::bitonic_sort_f32x8_desc;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cmp::{Ordering, Reverse};
// use std::collections::BinaryHeap;
// use std::ops::{AddAssign, Div, Sub};
use std::ptr;

/*
pub fn experts_topk_softmax_norm(
    input_ptr: *const f32,
    // [num_experts]
    experts_indicator_ptr: *mut bool,
    // token_size = sequence_chunk_size * batch_size
    // [num_experts, token_size]
    indices_ptr: *mut bool,
    value_ptr: *mut f32,
    index_token: usize,
    num_token: usize,
    num_experts: usize,
    num_topk: usize,
) {


}
*/

#[derive(Copy, Clone)]
struct HeapElem {
    value: f32,
    index: i32,
}
impl HeapElem {
    const NEG_INF: HeapElem = HeapElem {
        value: f32::NEG_INFINITY,
        index: -1,
    };
}
impl PartialEq for HeapElem {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.value.to_bits() == other.value.to_bits()
    }
}
impl Eq for HeapElem {}
impl PartialOrd for HeapElem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapElem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value
            .total_cmp(&other.value)
            .then_with(|| self.index.cmp(&other.index))
    }
}

struct FixedMinHeap<const CAP: usize> {
    data: [HeapElem; CAP],
    len: usize,
    limit: usize,
}
impl<const CAP: usize> FixedMinHeap<CAP> {
    fn new(limit: usize) -> Self {
        assert!(limit <= CAP);
        Self {
            data: [HeapElem::NEG_INF; CAP],
            len: 0,
            limit,
        }
    }
    fn len(&self) -> usize {
        self.len
    }
    fn push(&mut self, elem: HeapElem) {
        if self.limit == 0 {
            return;
        }
        if self.len < self.limit {
            self.data[self.len] = elem;
            self.sift_up(self.len);
            self.len += 1;
        } else if elem > self.data[0] {
            self.data[0] = elem;
            self.sift_down(0);
        }
    }
    fn sort_desc(&mut self) {
        self.data[..self.len].sort_unstable_by(|a, b| b.cmp(a));
    }
    fn as_slice(&self) -> &[HeapElem] {
        &self.data[..self.len]
    }
    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) >> 1;
            if self.data[idx] < self.data[parent] {
                self.data.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }
    fn sift_down(&mut self, mut idx: usize) {
        let len = self.len;
        loop {
            let left = (idx << 1) + 1;
            if left >= len {
                break;
            }
            let right = left + 1;
            let smallest = if right < len && self.data[right] < self.data[left] {
                right
            } else {
                left
            };
            if self.data[smallest] < self.data[idx] {
                self.data.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn get_topk(
    input_ptr: *const f32,
    len: usize,
    topk: usize,
    out_values: *mut f32,
    out_indices: *mut i32,
) {
    assert!(!input_ptr.is_null());
    assert!(!out_values.is_null());
    assert!(!out_indices.is_null());
    assert!(len % 8 == 0, "len must be divisible by 8");
    assert!(topk > 0 && topk <= 8, "topk must be within 1..=8");
    assert!(topk <= len, "topk cannot exceed len");

    let lane_offsets = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut heap = FixedMinHeap::<8>::new(topk);

    for chunk_start in (0..len).step_by(8) {
        let values = _mm256_loadu_ps(input_ptr.add(chunk_start));
        let base = _mm256_set1_epi32(chunk_start as i32);
        let indices = _mm256_add_epi32(base, lane_offsets);
        let (sorted_vals, sorted_idx) = bitonic_sort_f32x8_desc(values, indices);

        let mut chunk_vals = [0.0f32; 8];
        let mut chunk_idx = [0i32; 8];
        _mm256_storeu_ps(chunk_vals.as_mut_ptr(), sorted_vals);
        _mm256_storeu_si256(chunk_idx.as_mut_ptr() as *mut __m256i, sorted_idx);

        let chunk_take = topk.min(8);
        for lane in 0..chunk_take {
            heap.push(HeapElem {
                value: chunk_vals[lane],
                index: chunk_idx[lane],
            });
        }
    }

    debug_assert_eq!(heap.len(), topk);
    heap.sort_desc();
    for (i, elem) in heap.as_slice().iter().enumerate() {
        ptr::write(out_values.add(i), elem.value);
        ptr::write(out_indices.add(i), elem.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_get_topk() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let data = [
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        ];
        let mut out_vals = [0.0f32; 4];
        let mut out_idx = [0i32; 4];

        unsafe {
            get_topk(
                data.as_ptr(),
                data.len(),
                4,
                out_vals.as_mut_ptr(),
                out_idx.as_mut_ptr(),
            );
        }

        let mut expected: Vec<(f32, usize)> = data
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, val)| (val, idx))
            .collect();
        expected.sort_by(|a, b| b.0.total_cmp(&a.0));

        for i in 0..4 {
            assert_ulps_eq!(out_vals[i], expected[i].0);
            assert_eq!(out_idx[i], expected[i].1 as i32);
        }
    }

    #[test]
    fn test_fixed_min_heap_topk() {
        let samples = [1.5, -0.5, 2.25, 4.0, 3.5];
        let mut heap = FixedMinHeap::<8>::new(3);
        for (idx, &val) in samples.iter().enumerate() {
            heap.push(HeapElem {
                value: val,
                index: idx as i32,
            });
        }
        assert_eq!(heap.len(), 3);
        heap.sort_desc();
        let slice = heap.as_slice();
        let expected = [(4.0, 3), (3.5, 4), (2.25, 2)];
        for (elem, exp) in slice.iter().zip(expected.iter()) {
            assert_ulps_eq!(elem.value, exp.0);
            assert_eq!(elem.index, exp.1);
        }
    }
}
