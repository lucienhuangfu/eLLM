
pub trait MapTrait<T> {
    fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize);
}

pub trait SoftmaxTrait<T> {
    fn compute(&self, ptr1: *const T, indice_ptr: *mut T, value_ptr: *mut T, length: usize);
}

pub trait TopKSoftmaxTrait<T> {
    fn compute(&self, indices_ptr: *const T, values_ptr: *const T, sum_ptr: *const T, indice_ptr: *mut T, value_ptr: *mut T, length: usize);
}
