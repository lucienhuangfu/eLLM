pub trait MapTrait<T> {
    fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize);
}

pub trait SoftmaxTrait<T> {
    fn compute(
        &self,
        ptr1: *const T,
        experts_indicator_ptr: *mut bool,
        indice_ptr: *mut bool,
        weight_ptr: *mut T,
        token_index: usize,
        input_length: usize,
        output_length: usize,
    );
}

pub trait TopKSoftmaxTrait<T> {
    fn compute(
        &self,
        indices_ptr: *const usize,
        values_ptr: *const T,
        sum_ptr: *const T,
        indice_ptr: *mut usize,
        value_ptr: *mut T,
        length: usize,
    );
}
