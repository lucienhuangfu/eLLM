use std::alloc::{self, Layout};
use std::{mem, ops, ptr, slice};

/// Threshold for MADV_HUGEPAGE hint (glibc switches to mmap at ~128 KiB).
/// 触发 MADV_HUGEPAGE 提示的阈值（glibc 在 128 KiB 以上使用 mmap）。
const HUGEPAGE_HINT_THRESHOLD: usize = 128 * 1024;

/// Allocate a zero-initialised `Box<[T]>` via `alloc_zeroed`.
/// Much faster than `vec![T::default(); size]` because partial-panel
/// tails must be zero for the micro-kernel, and the OS can supply
/// pre-zeroed pages for large allocations.
/// 通过 `alloc_zeroed` 分配零初始化的 `Box<[T]>`。比逐元素标量循环
/// 快得多，因为 OS 可以为大块分配提供预零页。
pub fn alloc_zeroed_box<T>(size: usize) -> Box<[T]> {
    assert!(size > 0, "alloc_zeroed_box: size must be > 0");
    let layout = Layout::array::<T>(size)
        .unwrap_or_else(|_| panic!("alloc_zeroed_box: layout overflow for {size} T"));
    unsafe {
        let ptr = alloc::alloc_zeroed(layout) as *mut T;
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        hint_huge_page(ptr as *mut u8, layout.size());
        Box::from_raw(slice::from_raw_parts_mut(ptr, size))
    }
}

/// Hint the kernel to use transparent huge pages for this allocation.
/// 提示内核为此分配使用透明大页。
fn hint_huge_page(ptr: *mut u8, size_bytes: usize) {
    #[cfg(target_os = "linux")]
    if size_bytes >= HUGEPAGE_HINT_THRESHOLD {
        extern "C" {
            fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
        }
        // MADV_HUGEPAGE = 14 on Linux
        unsafe { madvise(ptr, size_bytes, 14); }
    }
    let _ = (ptr, size_bytes);
}

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
            hint_huge_page(ptr as *mut u8, layout.size());
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

    /// Zero-initialize via `write_bytes`. Only safe for types where
    /// all-zero-bytes equals `T::default()` (f16, f32, usize, etc.).
    /// 用 `write_bytes` 做零初始化，只适用于零字节等于默认值的类型
    /// （f16, f32, usize 等）。
    pub fn allocate_zero(length: usize) -> Self {
        let mut boxed = Self::allocate(length);
        unsafe {
            std::ptr::write_bytes(boxed.ptr as *mut u8, 0, length * mem::size_of::<T>());
        }
        boxed
    }

    /// Allocate without initializing. Caller must ensure every element
    /// is written before reading.
    /// 分配但不初始化。调用者必须保证每个元素在读之前被写入。
    pub fn allocate_uninit(length: usize) -> Self {
        Self::allocate(length)
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
