use std::cmp::Ordering;
use std::ptr;
#[derive(Clone)]
pub struct FixedMinHeap<T> {
    values: *mut T,
    indices: *mut usize,
    len: usize,
    limit: usize,
}

impl<T: PartialOrd + Copy> FixedMinHeap<T> {
    #[inline(always)]
    pub fn new(values: *mut T, indices: *mut usize, limit: usize) -> Self {
        debug_assert!(!values.is_null());
        debug_assert!(!indices.is_null());
        Self {
            values,
            indices,
            len: 0,
            limit,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn push(&mut self, value: T, index: usize) {
        if self.limit == 0 {
            return;
        }

        if self.len < self.limit {
            let idx = self.len;
            self.len += 1;
            self.sift_up_with(idx, value, index);
            return;
        }

        let root_value = self.value_at(0);
        let root_index = self.index_at(0);
        if Self::cmp_pair(root_value, root_index, value, index) == Ordering::Less {
            self.sift_down_with(0, value, index, self.len);
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn sort_desc(&mut self) {
        if self.len <= 1 {
            return;
        }

        for end in (1..self.len).rev() {
            self.swap(0, end);
            self.sift_down_range(0, end);
        }
    }

    #[inline(always)]
    fn set(&mut self, idx: usize, value: T, index: usize) {
        unsafe {
            ptr::write(self.values.add(idx), value);
            ptr::write(self.indices.add(idx), index);
        }
    }

    #[inline(always)]
    fn value_at(&self, idx: usize) -> T {
        unsafe { *self.values.add(idx) }
    }

    #[inline(always)]
    fn index_at(&self, idx: usize) -> usize {
        unsafe { *self.indices.add(idx) }
    }

    #[inline(always)]
    fn cmp_pair(value_a: T, index_a: usize, value_b: T, index_b: usize) -> Ordering {
        match value_a.partial_cmp(&value_b) {
            Some(ordering) if ordering != Ordering::Equal => ordering,
            _ => index_a.cmp(&index_b),
        }
    }

    #[inline(always)]
    fn cmp_at(&self, lhs: usize, rhs: usize) -> Ordering {
        Self::cmp_pair(
            self.value_at(lhs),
            self.index_at(lhs),
            self.value_at(rhs),
            self.index_at(rhs),
        )
    }

    #[inline(always)]
    fn swap(&mut self, a: usize, b: usize) {
        unsafe {
            ptr::swap(self.values.add(a), self.values.add(b));
            ptr::swap(self.indices.add(a), self.indices.add(b));
        }
    }

    #[inline(always)]
    fn sift_up_with(&mut self, mut idx: usize, value: T, index: usize) {
        while idx > 0 {
            let parent = (idx - 1) >> 1;
            let parent_value = self.value_at(parent);
            let parent_index = self.index_at(parent);
            if Self::cmp_pair(value, index, parent_value, parent_index) == Ordering::Less {
                self.set(idx, parent_value, parent_index);
                idx = parent;
            } else {
                break;
            }
        }
        self.set(idx, value, index);
    }

    #[inline(always)]
    fn sift_down_with(&mut self, mut idx: usize, value: T, index: usize, len: usize) {
        loop {
            let left = (idx << 1) + 1;
            if left >= len {
                break;
            }

            let right = left + 1;
            let mut child = left;
            if right < len && self.cmp_at(right, left) == Ordering::Less {
                child = right;
            }

            let child_value = self.value_at(child);
            let child_index = self.index_at(child);
            if Self::cmp_pair(child_value, child_index, value, index) == Ordering::Less {
                self.set(idx, child_value, child_index);
                idx = child;
            } else {
                break;
            }
        }

        self.set(idx, value, index);
    }

    #[inline(always)]
    fn sift_down_range(&mut self, idx: usize, len: usize) {
        if len <= 1 {
            return;
        }

        let value = self.value_at(idx);
        let index = self.index_at(idx);
        self.sift_down_with(idx, value, index, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_fixed_min_heap_topk() {
        let samples = [1.5, -0.5, 2.25, 4.0, 3.5];
        let mut values = [0.0f32; 3];
        let mut indices = [0usize; 3];

        let mut heap = FixedMinHeap::new(values.as_mut_ptr(), indices.as_mut_ptr(), values.len());
        for (idx, &val) in samples.iter().enumerate() {
            heap.push(val, idx);
        }
        assert_eq!(heap.len(), 3);
        heap.sort_desc();
        for (i, exp) in [(4.0, 3usize), (3.5, 4usize), (2.25, 2usize)]
            .iter()
            .enumerate()
        {
            assert_ulps_eq!(values[i], exp.0);
            assert_eq!(indices[i], exp.1);
        }
    }

    #[test]
    fn test_fixed_min_heap_empty() {
        let mut values = [0.0f32; 3];
        let mut indices = [0usize; 3];
        let heap = FixedMinHeap::new(values.as_mut_ptr(), indices.as_mut_ptr(), values.len());
        assert_eq!(heap.len(), 0);
    }

    #[test]
    fn test_fixed_min_heap_limit_zero() {
        let mut values = [0.0f32; 0];
        let mut indices = [0usize; 0];
        let mut heap = FixedMinHeap::new(values.as_mut_ptr(), indices.as_mut_ptr(), 0);
        heap.push(1.0, 0);
        assert_eq!(heap.len(), 0);
    }

    #[test]
    fn test_fixed_min_heap_partial_fill() {
        let mut values = [0.0f32; 5];
        let mut indices = [0usize; 5];
        let mut heap = FixedMinHeap::new(values.as_mut_ptr(), indices.as_mut_ptr(), 5);
        heap.push(1.0, 0);
        heap.push(2.0, 1);
        assert_eq!(heap.len(), 2);

        heap.sort_desc();
        assert_ulps_eq!(values[0], 2.0);
        assert_eq!(indices[0], 1);
        assert_ulps_eq!(values[1], 1.0);
        assert_eq!(indices[1], 0);
    }

    #[test]
    fn test_fixed_min_heap_duplicates() {
        let mut values = [0.0f32; 2];
        let mut indices = [0usize; 2];
        let mut heap = FixedMinHeap::new(values.as_mut_ptr(), indices.as_mut_ptr(), 2);

        heap.push(1.0, 0);
        heap.push(1.0, 1);
        heap.push(1.0, 2);

        assert_eq!(heap.len(), 2);
        heap.sort_desc();

        // Expect (1.0, 2) and (1.0, 1) because they are "larger" than (1.0, 0) due to index tie-breaking
        assert_ulps_eq!(values[0], 1.0);
        assert_eq!(indices[0], 2);
        assert_ulps_eq!(values[1], 1.0);
        assert_eq!(indices[1], 1);
    }

    #[test]
    fn test_fixed_min_heap_clear() {
        let mut values = [0.0f32; 3];
        let mut indices = [0usize; 3];
        let mut heap = FixedMinHeap::new(values.as_mut_ptr(), indices.as_mut_ptr(), 3);
        heap.push(1.0, 0);
        assert_eq!(heap.len(), 1);
        heap.clear();
        assert_eq!(heap.len(), 0);
        heap.push(2.0, 1);
        assert_eq!(heap.len(), 1);
        assert_ulps_eq!(values[0], 2.0);
        assert_eq!(indices[0], 1);
    }
}
