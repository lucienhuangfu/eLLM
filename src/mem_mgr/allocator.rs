use std::alloc::{self, Layout};
use std::{mem, ops, ptr, slice};

/// 对齐内存管理器，64字节对齐，适合SIMD512操作
#[derive(Debug)]
pub struct AlignedBox<T> {
    ptr: *mut T,
    length: usize,
    layout: Layout,
}

impl<T> AlignedBox<T> {
    pub fn allocate(length: usize) -> Self {
        assert!(length > 0, "Length must be greater than 0");

        unsafe {
            let layout = Layout::from_size_align_unchecked(length * mem::size_of::<T>(), 64);
            let ptr = alloc::alloc(layout) as *mut T;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            AlignedBox {
                ptr,
                length,
                layout,
            }
        }
    }

    pub fn allocate_init(length: usize, value: T) -> Self
    where
        T: Copy,
    {
        let mut boxed = Self::allocate(length);
        unsafe {
            let mut p = boxed.ptr;
            for _ in 0..length {
                ptr::write(p, value);
                p = p.add(1);
            }
        }
        boxed
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.length) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.length) }
    }

    #[inline]
    pub fn as_ptr_offset(&self, offset: usize) -> *const T {
        assert!(offset < self.length, "Offset out of bounds");
        unsafe { self.ptr.add(offset) }
    }

    #[inline]
    pub fn as_mut_ptr_offset(&self, offset: usize) -> *mut T {
        assert!(offset < self.length, "Offset out of bounds");
        unsafe { self.ptr.add(offset) }
    }
}

impl<T> ops::Deref for AlignedBox<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> ops::DerefMut for AlignedBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Drop for AlignedBox<T> {
    fn drop(&mut self) {
        unsafe {
            if mem::needs_drop::<T>() {
                for i in 0..self.length {
                    ptr::drop_in_place(self.ptr.add(i));
                }
            }
            alloc::dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

unsafe impl<T: Send> Send for AlignedBox<T> {}
unsafe impl<T: Sync> Sync for AlignedBox<T> {}

impl<T: Clone> Clone for AlignedBox<T> {
    fn clone(&self) -> Self {
        let mut cloned = Self::allocate(self.length);
        unsafe {
            ptr::copy_nonoverlapping(self.ptr, cloned.ptr, self.length);
        }
        cloned
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_aligned_box_allocate() {
        let length = 50;
        let mut boxed = AlignedBox::<f32>::allocate_init(length, 1.2);
        assert_eq!(boxed.as_ptr() as usize % 64, 0);
        for i in 0..length {
            assert_ulps_eq!(boxed[i], 1.2, max_ulps = 4);
        }
        assert_eq!(boxed.len(), length);
    }

    #[test]
    fn test_aligned_box_clone() {
        let length = 20;
        let mut boxed1 = AlignedBox::<usize>::allocate_init(length, 42);
        let boxed2 = boxed1.clone();
        for i in 0..length {
            assert_eq!(boxed1[i], boxed2[i]);
        }
    }
}
