use std::f16;
// use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    // matmul_params::MatmulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::Matmul5Trait;

// merge num_experts_per_tok个experts的结果
// 加上残差

#[derive(Clone)]
pub struct ExpertsMergeAdd<T> {
    input_ptr: ConstPtr<T>,
    residual_ptr: ConstPtr<T>,
    experts_indicator: MutPtr<bool>,
    indice_ptr: MutPtr<bool>,
    output_ptr: MutPtr<T>,
    batch_size: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    hidden_size: usize,
    // _marker: PhantomData<T>,
}
impl<T> ExpertsMergeAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn new(
        input_ptr: *const T, // TODO: Fix parameter name - should match struct field
        residual_ptr: *const T,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        output_ptr: *mut T,
        batch_size: usize,
        num_experts: usize,
        num_experts_per_token: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            residual_ptr: ConstPtr { ptr: residual_ptr },
            experts_indicator: MutPtr {
                ptr: experts_indicator,
            },
            indice_ptr: MutPtr { ptr: indice_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            batch_size,
            num_experts,
            num_experts_per_token,
            hidden_size,
            // _marker: PhantomData,
        }
    }

    pub fn run(
        &self,
        // position_index: usize,
        // position_interval: usize,
        batch_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        let num_tokens = self.batch_size;
        // 重置gate_routing数据结构
        if let Some((begin, end)) = assign(self.num_experts, thread_num, thread_id) {
            let experts_indicator_ptr = self.experts_indicator.ptr;
            let indices_ptr = self.indice_ptr.ptr;
            // println!("ExpertsMergeAdd reset from {} to {}", begin, end);

            for i in begin..end {
                // println!(   "ExpertsMergeAdd reset expert {}", i);
                unsafe {
                    if *experts_indicator_ptr.add(i) {
                        *experts_indicator_ptr.add(i) = false;
                        // reset indices_ptr using write_bytes for better performance
                        // write_bytes(p, 0, count) sets all bytes to 0, which means false for bool
                        let p = indices_ptr.add(i * num_tokens);
                        std::ptr::write_bytes(p, 0, num_tokens);
                    }
                }
            }
        }

        // TODO: Implement parallel matrix multiplication with experts merging
        // 1. Calculate chunk dimensions based on macro step sizes
        // 2. Determine work distribution for current thread
        // 3. Iterate through assigned chunks
        // 4. For each chunk, perform matrix multiplication of experts
        // 5. Merge results from multiple experts
        // 6. Add residual connection to final output
    }
}

impl<T> Matmul5Trait<T> for ExpertsMergeAdd<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(
        &self,
        input_ptr1: *const T, // First experts weights
        input_ptr2: *const T, // Second experts weights
        input_ptr3: *const T, // Third experts weights
        input_ptr4: *const T, // Fourth experts weights
        input_ptr5: *const T, // Fifth experts weights
        output_ptr: *mut T,   // Output buffer for merged results
    ) {

        // TODO: Generic implementation for merging 5 experts
        // 1. Perform matrix multiplication for each experts
        // 2. Apply experts routing weights/gates
        // 3. Sum weighted experts outputs
        // 4. Add residual connection from input

        // Use generic kernel for matrix multiplication
        // kernel::generic::matmul_5experts_merge::compute(
        //     input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5,
        //     self.input_ptr.ptr, self.residual_ptr.ptr, output_ptr,
        //     &self.params
        // );
    }
}

impl Matmul5Trait<f16> for ExpertsMergeAdd<f16> {
    fn compute(
        &self,
        input_ptr1: *const f16,
        input_ptr2: *const f16,
        input_ptr3: *const f16,
        input_ptr4: *const f16,
        input_ptr5: *const f16,
        output_ptr: *mut f16,
    ) {
        // TODO: Optimized f16 implementation using SIMD when available
        // Use AVX512FP16 instructions on x86_64 if available

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            // TODO: Call optimized AVX512FP16 kernel for 5-experts merge
            // kernel::x86_64::f16_512::experts_merge_add::compute_5experts(...)
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            // TODO: Fallback to generic implementation
            // Call generic compute method as fallback
        }
    }
}

impl Matmul5Trait<f32> for ExpertsMergeAdd<f32> {
    fn compute(
        &self,
        input_ptr1: *const f32,
        input_ptr2: *const f32,
        input_ptr3: *const f32,
        input_ptr4: *const f32,
        input_ptr5: *const f32,
        output_ptr: *mut f32,
    ) {
        // TODO: Optimized f32 implementation using AVX2 when available
        // Merge outputs from 5 experts with proper weighting

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            // TODO: Use SIMD instructions for parallel computation
            // kernel::x86_64::f32_256::experts_merge_add::compute_5experts(
            //     input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5,
            //     self.input_ptr.ptr, self.residual_ptr.ptr, output_ptr,
            //     &self.params
            // );
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            // TODO: Generic f32 implementation
            // Use generic matrix multiplication with experts merging
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use std::thread;

    /*
    #[test]
    fn test_f32_chunk() {
        let sequence_chunk_size = 8;
        let batch_size = 128;
        let a_row = batch_size;
        let b_row = 256;
        let column = 256;
        let a_row_step_macro = 32;
        let b_row_step_macro = 32;
        let column_step_macro = 64;
        let a_row_step_micro = 8;
        let b_row_step_micro = 8;

        let mut a = vec![0.0; a_row * column];
        let b = vec![1.0; b_row * column];
        // lay data into a portion of matrix a
        // fill the first batch_size rows with 1
        for i in 0..batch_size {
            for j in 0..column {
                a[i * column + j] = 1.0;
            }
        }
        let mut c = vec![0.0; a_row * b_row];
        let mut expected = vec![0.0; a_row * b_row];
        // calculate expected using the naive method
        for i in 0..batch_size {
            for j in 0..b_row {
                for k in 0..column {
                    expected[i * b_row + j] += a[i * column + k] * b[j * column + k];
                }
            }
        }

        // initialize the params
        let params: matmulParams = matmulParams {
            a_row,
            b_row,
            column,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
        };
        let mut operator = matmul::<f32>::new(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            // sequence_chunk_size,
            false,
            a_row,
            b_row,
            column,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
        );
        let thread_num: usize = num_cpus::get();

        for i in 0..thread_num {
            // println!("{}", i);
            operator.run(0, sequence_chunk_size, batch_size, thread_num, i);
        }


        // assert_eq!(c, expected);

        // print the result
        for i in 0..a_row {
            for j in 0..b_row {
                //print!("{:?} ", c[i * b_row + j]);
            }
            //println!();
        }
    }
     */
}
