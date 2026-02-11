use std::cell::SyncUnsafeCell;

// define a struct for storing the parameters of the matrix multiplication
pub struct ConstPtr<T> {
    pub ptr: *const T,
}
unsafe impl<T> Sync for ConstPtr<T> {}
unsafe impl<T> Send for ConstPtr<T> {}

impl<T> Copy for ConstPtr<T> {}

impl<T> Clone for ConstPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

pub struct MutPtr<T> {
    pub ptr: *mut T,
}
unsafe impl<T> Sync for MutPtr<T> {}
unsafe impl<T> Send for MutPtr<T> {}

impl<T> Copy for MutPtr<T> {}

impl<T> Clone for MutPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

pub struct SharedMut<T> {
    cell: SyncUnsafeCell<T>,
}

impl<T> SharedMut<T> {
    pub fn new(value: T) -> Self {
        Self {
            cell: SyncUnsafeCell::new(value),
        }
    }

    pub fn get(&self) -> *mut T {
        self.cell.get()
    }
}

unsafe impl<T> Sync for SharedMut<T> {}
unsafe impl<T> Send for SharedMut<T> {}
