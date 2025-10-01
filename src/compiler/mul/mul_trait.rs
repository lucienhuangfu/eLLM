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

// === runner/mul_trait.rs ===
pub trait MatMul3Trait<T> {
    fn compute1(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        input_ptr3: *const T,
        output_ptr1: *mut T,
        output_ptr2: *mut T,
    );

    fn compute2(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
    );
}
pub trait MatMul4Trait<T> {
    /// 内核1：C += A × B_panel（3x128 累加，按 kc 反复进入）
    fn compute1(&self, a: *const T, b_panel: *const T, c: *mut T);

    /// 内核2：对 3x128 tile 做收尾（RMSNorm(weight=1) + RoPE），就地写回 C
    /// rope_ptr 指向交错 [cos0,sin0,cos1,sin1,...] 的 head_dim(=128) 相位行
    fn compute2(&self, c: *mut T, rope_ptr: *const T);
}