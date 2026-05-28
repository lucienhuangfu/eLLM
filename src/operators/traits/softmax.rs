pub trait MatMulTopKTrait<T> {
    fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, value_ptr: *mut T);
}

pub trait TopKSoftmaxTrait<T> {
    fn compute(
        &self,
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        temperature: T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        thread_num: usize,
        top_k: usize,
        top_k_simd: usize,
    );
}

pub trait SoftmaxTrait<T> {
    fn compute(
        &self,
        ptr1: *const T,
        topk_values_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        input_length: usize,
        output_length: usize,
    );
}

pub trait ExpertTopkNormTrait<T> {
    fn compute(
        &self,
        ptr1: *const T,
        topk_values_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        input_length: usize,
        output_length: usize,
    );
}
