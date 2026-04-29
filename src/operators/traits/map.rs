pub trait ZipMapTrait<T> {
    fn compute(&self, input_ptr1: *const T, input_ptr_2: *const T, output_ptr: *mut T);
}

pub trait MapTrait<T> {
    fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize);
}
