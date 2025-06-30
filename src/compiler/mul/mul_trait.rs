pub trait MatlMulTrait<T> {
    fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T);
    fn compute2(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T, length: usize);
}

pub trait AttentionMulTrait<T> 
// where T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialOrd + NegInfinity + Exp,
{
    fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        input_ptr3: *const T,
        output_ptr: *mut T,
        position: usize,
    );
}