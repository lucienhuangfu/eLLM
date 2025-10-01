
pub trait MapTrait<T> {
    fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize);
}

pub trait TopKSoftmaxTrait<T> {
    fn compute(&self, indice_ptr: *const T, value_ptr: *const T, sum_ptr: *const T, output_ptr: *mut T, length: usize);
}
