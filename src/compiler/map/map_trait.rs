
pub trait MapTrait<T> {
    fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize);
}

pub trait SoftmaxTrait<T> {
    fn compute(&self, input_ptr: *const T, sum_ptr: *const T, max_ptr: *const T, output_ptr: *mut T, length: usize);
}
