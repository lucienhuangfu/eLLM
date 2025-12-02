use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatmulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::Matmul2Trait;
// 完成down projection的Matmul
// 乘以weight
// 然后根据sorted_ids把结果放到对应的位置 [batch_size, num_experts_per_tok,  hidden_size]

#[derive(Clone)]
pub struct ExpertsMatmulMul<T> {
    input_ptr: ConstPtr<T>,
    down_weight_ptr: ConstPtr<T>,
    // Expert routing information
    // sorted [num_experts, [(token_index, weight)]]
    // [num_experts]
    experts_indicator: MutPtr<bool>,
    // [num_experts, batch_size]
    indice_ptr: MutPtr<bool>,
    // [num_experts, batch_size]
    weight_ptr: MutPtr<T>,
    topk_indices_ptr: MutPtr<usize>,
    output_ptr: MutPtr<T>,
    a_row: usize,
    b_row: usize,
    column: usize,
    pub params: MatmulParams,
    decode_only_flag: bool,
    _marker: PhantomData<T>,
}
impl<T> ExpertsMatmulMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn new(
        input_ptr: *const T,
        down_weight_ptr: *const T,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        weight_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        output_ptr: *mut T,
        a_row: usize,
        b_row: usize,
        column: usize,
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize,
        b_row_step_micro: usize,
        decode_only_flag: bool,
    ) -> Self {
        Self {
            input_ptr: ConstPtr { ptr: input_ptr },
            down_weight_ptr: ConstPtr {
                ptr: down_weight_ptr,
            },
            experts_indicator: MutPtr { ptr: experts_indicator },
            indice_ptr: MutPtr { ptr: indice_ptr },
            weight_ptr: MutPtr { ptr: weight_ptr },
            topk_indices_ptr: MutPtr { ptr: topk_indices_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            a_row,
            b_row,
            column,
            params: MatmulParams {
                a_row_step_macro,
                b_row_step_macro,
                column_step_macro,
                a_row_step_micro,
                b_row_step_micro,
            },
            decode_only_flag,
            _marker: PhantomData,
        }
    }

    pub fn run(
        &self,
        // position_index: usize,
        // position_interval: usize,
        batch_size: usize,
        decode_size: usize, 
        thread_num: usize,
        thread_id: usize,
    ) {
        /*
        let (mut a_chunk_num, remainder) = (self.params.a_row / self.params.a_row_step_macro, self.params.a_row % self.params.a_row_step_macro);
        if remainder > 0 {
            a_chunk_num += 1;
        }
        let b_chunk_num = self.params.b_row / self.params.b_row_step_macro;
        if let Some((begin, end)) = assign(position_interval * a_chunk_num * b_chunk_num, cpu_num, thread_id)
        {
            //  [sequence_chunk_size, batch_size, hidden_size]
            let (mut row_index, mut col_index) = (begin / batch_size, begin % batch_size);
                          let ptr1 = if self.output_to_kv {
        self.ptr1.ptr.add(position_begin * max_stride * self.head_size)
                } else {
                    self.ptr1.ptr
                };
            let mut input_ptr2 = self.ptr2.ptr;
            let mut output_ptr = self.output_ptr.ptr;
        }*/



    }
}

impl<T> Matmul2Trait<T> for ExpertsMatmulMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(
        &self,
        input_ptr: *const T,
        down_weight_ptr: *const T,
        output_ptr: *mut T,
        weight: T,
    ) {
        /*
        //print!("generic runner\n");
        kernel::generic::Matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &(self.params),
        );*/
    }
}

impl Matmul2Trait<f16> for ExpertsMatmulMul<f16> {
    fn compute(
        &self,
        input_ptr: *const f16,
        down_weight_ptr: *const f16,
        output_ptr: *mut f16,
        weight: f16,
    ) {
        // print!("f16 runner\n");

        /*
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
        ); */
    }
}

impl Matmul2Trait<f32> for ExpertsMatmulMul<f32> {
    fn compute(
        &self,
        input_ptr: *const f32,
        down_weight_ptr: *const f32,
        output_ptr: *mut f32,
        weight: f32,
    ) {
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
