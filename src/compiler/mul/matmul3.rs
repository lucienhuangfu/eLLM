use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::MatMulTrait;

// there will be just one instance of this runner in the program
// this runner will be shared by many threads that together compute the matrix multiplication
#[derive(Clone)]
pub struct MatMul3<T> {
    hidden_ptr: ConstPtr<T>,
    q_weight_ptr: ConstPtr<T>,
    q_state_ptr: MutPtr<T>,
    k_weight_ptr: ConstPtr<T>,
    k_state_ptr: MutPtr<T>,
    v_weight_ptr: ConstPtr<T>,
    v_state_ptr: MutPtr<T>,
    position_embedding_ptr: ConstPtr<T>,
    head_dim: usize,
    a_h_row: usize,
    col: usize,
    b_q_row: usize,
    b_k_row: usize,
    b_v_row: usize,
    pub params: MatMulParams,
    _marker: PhantomData<T>,
}
impl<T> MatMul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn new(
        hidden_ptr: *const T,
        q_weight_ptr: *const T,
        q_state_ptr: *mut T,
        k_weight_ptr: *const T,
        k_state_ptr: *mut T,
        v_weight_ptr: *const T,
        v_state_ptr: *mut T,
        position_embedding_ptr: *const T,
        head_dim: usize,
        a_h_row: usize,
        col: usize,
        b_q_row: usize,
        b_k_row: usize,
        b_v_row: usize,
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize,
        b_row_step_micro: usize,
    ) -> Self {
        Self {
            hidden_ptr: ConstPtr { ptr: hidden_ptr },
            q_weight_ptr: ConstPtr { ptr: q_weight_ptr },
            q_state_ptr: MutPtr { ptr: q_state_ptr },
            k_weight_ptr: ConstPtr { ptr: k_weight_ptr },
            k_state_ptr: MutPtr { ptr: k_state_ptr },
            v_weight_ptr: ConstPtr { ptr: v_weight_ptr },
            v_state_ptr: MutPtr { ptr: v_state_ptr },
            position_embedding_ptr: ConstPtr {
                ptr: position_embedding_ptr,
            },
            head_dim: head_dim,
            a_h_row: a_h_row,
            col: col,
            b_q_row: b_q_row,
            b_k_row: b_k_row,
            b_v_row: b_v_row,
            params: MatMulParams {
                // a_row,
                // b_row,
                // column,
                a_row_step_macro,
                b_row_step_macro,
                column_step_macro,
                a_row_step_micro,
                b_row_step_micro,
            },
            _marker: PhantomData,
        }
    }

    pub fn run(
        &self,
        position_index: usize,
        position_interval: usize,
        batch_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        // norm的weight都是1，相当于没有weight,不需要乘以weight
        // c小块的列数需要是head_dim，以便处理norm

        // key: linear

        // value: linear -> norm -> complex

        // query: linear -> norm -> complex
    }
}

impl<T> MatMulTrait<T> for MatMul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        //print!("generic runner\n");
        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &(self.params),
        );
    }

    default fn compute2(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
        length: usize,
    ) {
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

impl MatMulTrait<f16> for MatMul3<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        // print!("f16 runner\n");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::matmul_block::matmul_block(
                input_ptr1,
                input_ptr2,
                output_ptr,
                &self.params,
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &(self.params),
        );
    }

    fn compute2(
        &self,
        input_ptr1: *const f16,
        input_ptr2: *const f16,
        output_ptr: *mut f16,
        length: usize,
    ) {
        // print!("f16 runner\n");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::dot_product::dot_product(
                input_ptr1, input_ptr2, output_ptr, length,
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

impl MatMulTrait<f32> for MatMul3<f32> {
    fn compute(&self, input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32) {
        // print!("f32 runner\n");

        /*//implementation for f32 on platform with avx2
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_matmul_block(a, b, c, param, a_row_l, b_row_l, column_l);
        };
        // generic implementation for f32
        // #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        // generic_matmul_block(input_ptr1, input_ptr2, output_ptr, &(self.params));
    }

    fn compute2(
        &self,
        input_ptr1: *const f32,
        input_ptr2: *const f32,
        output_ptr: *mut f32,
        length: usize,
    ) {
        // print!("f32 runner\n");

        /*//implementation for f32 on platform with avx2
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_matmul_block(a, b, c, param, a_row_l, b_row_l, column_l);
        };
        // generic implementation for f32
        // #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }
}

#[cfg(test)]
mod tests {
    use super::*;





}
