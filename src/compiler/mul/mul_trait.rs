pub trait MatMulTrait<T> {
    fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T);
    fn compute2(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T, length: usize);
}

pub trait AttentionMulAddTrait<T> 
// where T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialOrd + NegInfinity + Exp,
{
    fn compute(
        &self,
        q_ptr1: *const T,
        k_ptr2: *const T,
        v_ptr3: *const T,
        residual_ptr: *const T,
        output_ptr: *mut T,
        position: usize,
    );
}

pub trait MatMul3Trait<T> {
    fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, input_ptr3: *const T, output_ptr: *mut T);
}

pub trait MatMul4Trait<T> {
    fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, input_ptr3: *const T, output_ptr: *mut T);
}