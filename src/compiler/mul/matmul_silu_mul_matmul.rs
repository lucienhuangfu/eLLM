
use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};
use super::super::super::init::{
    matmul_params::MatMulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::MatMul3Trait;

// there will be just one instance of this runner in the program
// this runner will be shared by many threads that together compute the matrix multiplication
#[derive(Clone)]
pub struct MatMul3<T> {
    ptr1: ConstPtr<T>,
    ptr2: ConstPtr<T>,
    ptr3: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    pub params: MatMulParams,
    _marker: PhantomData<T>,
}
impl<T> MatMul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn new(
        ptr1: *const T,
        ptr2: *const T,
        ptr3: *const T,
        output_ptr: *mut T,
        // sequence_length: usize,
        // output_to_kv: bool,

        // these are the parameters of the matrix multiplication, this matrix is a largest possible one
        // for later matrix multiplication, the actual size of the matrix will be smaller
        // so this is reserving enough spaces in memory, and later lay the data into a small portion of it
        // and as we compute, we just access and calculate with the data in the small portion
        // this is like we construct a big playground, and we only play in a small or big portion of it, depending on how many people there are
        // so these dimensions are the dimensions of the largest possible matrix
        a_row: usize,
        b_row: usize,
        column: usize,
        // these are the sizes of the macro kernels
        // how they are determined is not clear
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        // these are the sizes of the micro kernels
        // how they are determined is not clear
        a_row_step_micro: usize,
        b_row_step_micro: usize,
    ) -> Self {
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: ptr2 },
            ptr3: ConstPtr { ptr: ptr3 },
            output_ptr: MutPtr { ptr: output_ptr },
            // sequence_length: sequence_length,
            // output_to_kv: output_to_kv,
            params: MatMulParams {
                a_row,
                b_row,
                column,
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

impl<T> MatMul3Trait<T> for MatMul3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T,input_ptr3: *const T, output_ptr: *mut T) {
        //print!("generic runner\n");
 
    }
}

impl MatMul3Trait<f16> for MatMul3<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, input_ptr3: *const f16, output_ptr: *mut f16) {
        // print!("f16 runner\n");


    }


}

impl MatMul3Trait<f32> for MatMul3<f32> {
    fn compute(&self, input_ptr1: *const f32, input_ptr2: *const f32, input_ptr3: *const f32, output_ptr: *mut f32) {
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

    // use super::super::chunk_matmul::chunk_matmul;

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
        let params: MatMulParams = MatMulParams {
            a_row,
            b_row,
            column,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
        };
        let mut operator = MatMul::<f32>::new(
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

   


}
