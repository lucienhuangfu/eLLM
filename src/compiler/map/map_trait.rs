
pub trait MapTrait<T> {
    fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize);
}
