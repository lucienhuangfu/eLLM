use std::alloc::{self, Layout};
use std::sync::OnceLock;
use std::{mem, ops, ptr, slice};

/// Threshold for MADV_HUGEPAGE hint (glibc switches to mmap at ~128 KiB).
/// 触发 MADV_HUGEPAGE 提示的阈值（glibc 在 128 KiB 以上使用 mmap）。
const HUGEPAGE_HINT_THRESHOLD: usize = 128 * 1024;

/// Anonymous mappings are demand-zero: creating the mapping does not fault in
/// and clear every page. Keep small buffers on the allocator because a mmap
/// syscall costs more than eagerly clearing a handful of pages.
const DEMAND_ZERO_THRESHOLD: usize = 128 * 1024;

/// Keep an opt-in for A/B measurements against the default eager
/// `allocate + write_bytes` behavior.
fn demand_zero_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ELLM_DEMAND_ZERO")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    })
}

#[derive(Debug)]
enum Allocation {
    Alloc(Layout),
    Mmap(usize),
}

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
        unsafe {
            madvise(ptr, size_bytes, 14);
        }
    }
    let _ = (ptr, size_bytes);
}

/// 对齐内存管理器，64字节对齐，适合SIMD512操作
#[derive(Debug)]
pub struct AlignedBox<T> {
    ptr: *mut T,
    length: usize,
    allocation: Allocation,
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
                allocation: Allocation::Alloc(layout),
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

    /// Zero-initialize without eagerly touching large allocations. Only safe for types where
    /// all-zero-bytes equals `T::default()` (f16, f32, usize, etc.).
    /// On Linux, anonymous mmap lets the kernel provide zero pages on demand;
    /// small allocations and other platforms use `alloc_zeroed`.
    /// 大块内存在 Linux 上使用匿名 mmap，依赖内核按需提供零页，避免分配时
    /// 触碰全部页面；小块内存仍使用 `alloc_zeroed`。
    pub fn allocate_zero(length: usize) -> Self {
        Self::allocate_zero_with_mode(length, demand_zero_enabled())
    }

    /// Allocate a large, short-lived staging buffer from an independent OS
    /// mapping. Unlike allocator arenas, dropping it reliably returns the
    /// resident pages even when allocation and destruction happen on different
    /// threads. The returned bytes are zeroed by the OS, but callers may
    /// immediately overwrite them.
    pub fn allocate_transient(length: usize) -> Self {
        Self::allocate_zero_with_mode(length, true)
    }

    fn allocate_zero_with_mode(length: usize, demand_zero: bool) -> Self {
        assert!(length > 0, "Length must be greater than 0");

        // This is the pre-optimization implementation, retained as a benchmark
        // control. It intentionally touches every page.
        if !demand_zero {
            let boxed = Self::allocate(length);
            unsafe {
                ptr::write_bytes(boxed.ptr as *mut u8, 0, length * mem::size_of::<T>());
            }
            return boxed;
        }

        let size_bytes = length
            .checked_mul(mem::size_of::<T>())
            .expect("AlignedBox allocation size overflow");
        let layout =
            Layout::from_size_align(size_bytes, 64).expect("AlignedBox allocation layout overflow");

        #[cfg(target_os = "linux")]
        if layout.size() >= DEMAND_ZERO_THRESHOLD {
            const PROT_READ: i32 = 0x1;
            const PROT_WRITE: i32 = 0x2;
            const MAP_PRIVATE: i32 = 0x02;
            const MAP_ANONYMOUS: i32 = 0x20;
            const MAP_FAILED: *mut std::ffi::c_void = !0usize as *mut std::ffi::c_void;
            extern "C" {
                fn mmap(
                    addr: *mut std::ffi::c_void,
                    length: usize,
                    prot: i32,
                    flags: i32,
                    fd: i32,
                    offset: isize,
                ) -> *mut std::ffi::c_void;
            }

            let ptr = unsafe {
                mmap(
                    ptr::null_mut(),
                    layout.size(),
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if ptr != MAP_FAILED {
                hint_huge_page(ptr as *mut u8, layout.size());
                return Self {
                    ptr: ptr as *mut T,
                    length,
                    allocation: Allocation::Mmap(layout.size()),
                };
            }
        }

        unsafe {
            let ptr = alloc::alloc_zeroed(layout) as *mut T;
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            hint_huge_page(ptr as *mut u8, layout.size());
            Self {
                ptr,
                length,
                allocation: Allocation::Alloc(layout),
            }
        }
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
            match self.allocation {
                Allocation::Alloc(layout) => alloc::dealloc(self.ptr as *mut u8, layout),
                Allocation::Mmap(size) => {
                    #[cfg(target_os = "linux")]
                    {
                        extern "C" {
                            fn munmap(addr: *mut std::ffi::c_void, length: usize) -> i32;
                        }
                        let result = munmap(self.ptr as *mut std::ffi::c_void, size);
                        debug_assert_eq!(result, 0, "munmap failed");
                    }
                    #[cfg(not(target_os = "linux"))]
                    unreachable!("mmap allocation is Linux-only");
                }
            }
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

    #[test]
    fn test_large_allocate_zero() {
        let length = DEMAND_ZERO_THRESHOLD / mem::size_of::<u64>() + 1;
        let boxed = AlignedBox::<u64>::allocate_zero_with_mode(length, true);

        assert_eq!(boxed.as_ptr() as usize % 64, 0);
        assert_eq!(boxed[0], 0);
        assert_eq!(boxed[length / 2], 0);
        assert_eq!(boxed[length - 1], 0);
        #[cfg(target_os = "linux")]
        assert!(matches!(boxed.allocation, Allocation::Mmap(_)));
    }
}
