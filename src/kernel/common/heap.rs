use std::cmp::Ordering;
use std::ptr;
#[derive(Clone)]

pub struct FixedMinHeap<T: PartialOrd + Copy> {
    values: *mut T,
    indices: *mut usize,
    len: usize,
    limit: usize,
}

impl<T: PartialOrd + Copy> FixedMinHeap<T> {
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

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push(&mut self, value: T, index: usize) {
        if self.limit == 0 {
            return;
        }
        unsafe {
            if self.len < self.limit {
                self.set(self.len, value, index);
                self.sift_up(self.len);
                self.len += 1;
            } else if Self::cmp_pair(self.value_at(0), self.index_at(0), value, index)
                == Ordering::Less
            {
                self.set(0, value, index);
                self.sift_down(0);
            }
        }
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn sort_desc(&mut self) {
        unsafe {
            for i in 0..self.len {
                let mut max = i;
                for j in (i + 1)..self.len {
                    if Self::cmp_pair(
                        self.value_at(j),
                        self.index_at(j),
                        self.value_at(max),
                        self.index_at(max),
                    ) == Ordering::Greater
                    {
                        max = j;
                    }
                }
                if max != i {
                    self.swap(i, max);
                }
            }
        }
    }

    fn set(&mut self, idx: usize, value: T, index: usize) {
        unsafe {
            ptr::write(self.values.add(idx), value);
            ptr::write(self.indices.add(idx), index);
        }
    }

    fn value_at(&self, idx: usize) -> T {
        unsafe { *self.values.add(idx) }
    }

    fn index_at(&self, idx: usize) -> usize {
        unsafe { *self.indices.add(idx) }
    }

    fn cmp_pair(value_a: T, index_a: usize, value_b: T, index_b: usize) -> Ordering {
        match value_a.partial_cmp(&value_b) {
            Some(ordering) if ordering != Ordering::Equal => ordering,
            _ => index_a.cmp(&index_b),
        }
    }

    fn cmp_at(&self, lhs: usize, rhs: usize) -> Ordering {
        Self::cmp_pair(
            self.value_at(lhs),
            self.index_at(lhs),
            self.value_at(rhs),
            self.index_at(rhs),
        )
    }

    fn swap(&mut self, a: usize, b: usize) {
        unsafe {
            ptr::swap(self.values.add(a), self.values.add(b));
            ptr::swap(self.indices.add(a), self.indices.add(b));
        }
    }

    fn sift_up(&mut self, mut idx: usize) {
        unsafe {
            while idx > 0 {
                let parent = (idx - 1) >> 1;
                if self.cmp_at(idx, parent) == Ordering::Less {
                    self.swap(idx, parent);
                    idx = parent;
                } else {
                    break;
                }
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        unsafe {
            loop {
                let left = (idx << 1) + 1;
                if left >= self.len {
                    break;
                }
                let right = left + 1;
                let mut smallest = left;
                if right < self.len && self.cmp_at(right, left) == Ordering::Less {
                    smallest = right;
                }
                if self.cmp_at(smallest, idx) == Ordering::Less {
                    self.swap(idx, smallest);
                    idx = smallest;
                } else {
                    break;
                }
            }
        }
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
}
}
