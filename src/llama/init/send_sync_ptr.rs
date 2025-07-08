// define a struct for storing the parameters of the matrix multiplication
#[derive( Copy)]
pub struct ConstPtr<T> {
    pub ptr: *const T,
}
unsafe impl<T> Sync for ConstPtr<T> {}
unsafe impl<T> Send for ConstPtr<T> {}
// Manually implement Copy for ConstPtr<T>
impl<T> Copy for ConstPtr<T> {}

// Manually implement Clone for ConstPtr<T> to ensure it works properly with Copy
impl<T> Clone for ConstPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[derive( Copy)]
pub struct MutPtr<T> {
    pub ptr: *mut T,
}
unsafe impl<T> Sync for MutPtr<T> {}
unsafe impl<T> Send for MutPtr<T> {}
// Manually implement Copy for MutPtr<T>
impl<T> Copy for MutPtr<T> {}

// Manually implement Clone for MutPtr<T> to ensure it works properly with Copy
impl<T> Clone for MutPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
