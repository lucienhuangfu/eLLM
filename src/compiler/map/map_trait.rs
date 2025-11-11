pub trait MapTrait<T> {
    fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize);
}

pub trait SoftmaxTrait<T> {
    fn compute(
        &self,
        ptr1: *const T,
        topk_values_ptr: *mut T,
        topk_indices_ptr: *mut usize,
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
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        sums_ptr: *const T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        output_token_ptr: *mut usize,
        thread_num: usize,
        topk_size: usize
    );
}
