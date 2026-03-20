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
        topk_size: usize,
    );
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

pub trait ExpertsSigmoidGateTrait<T> {
    fn compute(&self, m0: usize, n0: usize, m_blk: usize, n_blk: usize, thread_id: usize);
}

pub trait ExpertsTopkNormTrait<T> {
    fn compute(
        &self,
        ptr1: *const T,
        topk_values_ptr: *mut T,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        value_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        token_index: usize,
        batch_size: usize,
        input_length: usize,
        output_length: usize,
    );
}
