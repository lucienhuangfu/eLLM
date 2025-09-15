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
pub struct MatMul<T> {
    hidden_ptr: ConstPtr<T>,
    q_weight_ptr: ConstPtr<T>,
    q_state_ptr: ConstPtr<T>,
    k_weight_ptr: ConstPtr<T>,
    k_state_ptr: ConstPtr<T>,
    v_weight_ptr: ConstPtr<T>,
    v_state_ptr: ConstPtr<T>,
    a_h_row: usize,
    col: usize,
    b_q_row: usize,
    b_k_row: usize,
    b_v_row: usize,
    pub params: MatMulParams,
    _marker: PhantomData<T>,
}
impl<T> MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn new(
        hidden_ptr: *const T,
        q_weight_ptr: *const T,
        q_state_ptr: *const T,
        k_weight_ptr: *const T,
        k_state_ptr: *const T,
        v_weight_ptr: *const T,
        v_state_ptr: *const T,
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
            q_state_ptr: ConstPtr { ptr: q_state_ptr },
            k_weight_ptr: ConstPtr { ptr: k_weight_ptr },
            k_state_ptr: ConstPtr { ptr: k_state_ptr },
            v_weight_ptr: ConstPtr { ptr: v_weight_ptr },
            v_state_ptr: ConstPtr { ptr: v_state_ptr },
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

        // value: linear -> norm -> complex

        // key: linear

        // query: linear -> norm -> complex
    }
}

impl<T> MatMulTrait<T> for MatMul<T>
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

impl MatMulTrait<f16> for MatMul<f16> {
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

impl MatMulTrait<f32> for MatMul<f32> {
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

    /*
    // test f32 runner
    #[test]
    fn test_f32_to_kv() {
        let batch_size = 1;
        let sequence_length = 8;
        let position_index = 4;

        let a_row = 128;
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
        let mut c = vec![0.0; sequence_length * a_row * b_row];
        let mut expected = vec![0.0; sequence_length * a_row * b_row];
        let offset = a_row * b_row * position_index;
        // calculate expected using the naive method
        for i in 0..batch_size {
            for j in 0..b_row {
                for k in 0..column {
                    expected[i * b_row + j + offset] += a[i * column + k] * b[j * column + k];
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

        // get the tasks
        let chunks = chunk_matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &params);
        // initialize the runner and multi-threaded run
        // create as many threads as the logical cores in the machine
        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let mut runner = MatMul::<f32>::new(
            a_row,
            b_row,
            column,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
            sequence_length,
            thread_num,
            barrier_arc,
        );
        runner.set_chunk(chunks);
        let runner_arc = Arc::new(runner);

        thread::scope(|s| {
            let mut handles = Vec::with_capacity(thread_num);
            for thread_id in (0..thread_num) {
                let _thread_id = thread_id;
                let runner_arc_clone = Arc::clone(&runner_arc);

                let handle = s.spawn(move || {
                    runner_arc_clone.run(batch_size, position_index, _thread_id);
                });
                handles.push(handle);
            }
            for handle in handles {
                handle.join().unwrap();
            }
        });
        assert_eq!(c, expected);

        // print the result
        for i in 0..a_row {
            for j in 0..b_row {
                //print!("{:?} ", c[i * b_row + j]);
            }
            //println!();
        }
    }
    */

    /*
    // Helper function to compare two f16 arrays with a tolerance
    fn compare_f16_arrays(arr1: &[f16], arr2: &[f16], tolerance: f32, length: usize) -> bool {
        // tell if the first length elements in both arrays are equal within the tolerance
        /*if arr1.len() != arr2.len() {
            println!("length not equal");
            return false;
        }*/
        for i in 0..length {
            // let diff = (f32::from(arr1[i]) - f32::from(arr2[i])).abs();
            let diff = (arr1[i] - arr2[i]).abs();
            if diff > tolerance {
                //println!("diff: {}, a: {}, b: {}", diff, arr1[i], arr2[i]);
                return false;
            }
        }
        true
    }


    // test f16 runner
    #[test]
    fn test_f16_runner() {
        let batch_size = 1;
        let a_row = 128;
        let b_row = 128;
        let column = 128;
        let a_row_step_macro = 16;
        let b_row_step_macro = 16;
        let column_step_macro = 16;
        let a_row_step_micro = 8;
        let b_row_step_micro = 8;

        let mut a = vec![0.0; a_row * column];
        let b = vec![1.0; b_row * column];
        // lay data into a portion of matrix a
        // fill the first batch_size rows with 1
        for i in 0..batch_size*column {
            a[i] = 1.0;
        }
        // print matrix a
        //println!("matrix a: ");
        for i in 0..a_row {
            for j in 0..column {
                //print!("{:?} ", a[i * column + j]);
            }
            //println!();
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
        // get the tasks
        let chunks = chunk_matmul(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            &params,
        );
        // initialize the runner and multi-threaded run
        // create as many threads as the logical cores in the machine
        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let mut runner = MatMul::<f16>::new(
            a_row,
            b_row,
            column,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
            //chunks,
            1,
            thread_num,
            barrier_arc,
            // place total thread number here
        );
        // let runner_arc = Arc::new(runner);
        runner.set_chunk(chunks);
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(thread_num);
            for thread_id in (0..thread_num) {
                let _thread_id = thread_id;
                // let runner_arc_clone = Arc::clone(&runner_arc);
                let _runner = &runner;
                let handle = s.spawn(move || {
                    _runner.run(batch_size, thread_num, _thread_id);
                });
                handles.push(handle);
            }
            for handle in handles {
                handle.join().unwrap();
            }
        });
        // print the result
        for i in 0..batch_size {
            for j in 0..b_row {
                //print!("{:?} ", c[i * b_row + j]);
            }
            //println!();
        }

        assert!(compare_f16_arrays(&c, &expected, 1e-3, batch_size * b_row));


    }

    // test f32 runner
    #[test]
    fn test_f32_runner() {
        let batch_size = 30;
        let a_row = 128;
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
        // get the tasks
        let chunks = chunk_matmul(
            a.as_ptr(),

            b.as_ptr(),

            c.as_mut_ptr(),

            &params,
        );
        // initialize the runner and multi-threaded run
        // create as many threads as the logical cores in the machine
        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let mut runner = MatMul::<f32>::new(
            a_row,
            b_row,
            column,
            a_row_step_macro,
            b_row_step_macro,
            column_step_macro,
            a_row_step_micro,
            b_row_step_micro,
            1,
            thread_num,
            barrier_arc,
        );
        runner.set_chunk(chunks);
        let runner_arc = Arc::new(runner);

        thread::scope(|s| {
            let mut handles = Vec::with_capacity(thread_num);
            for thread_id in (0..thread_num) {
                let _thread_id = thread_id;
                let runner_arc_clone = Arc::clone(&runner_arc);

                let handle = s.spawn(move || {
                    runner_arc_clone.run(batch_size, thread_num, _thread_id);
                });
                handles.push(handle);
            }
            for handle in handles {
                handle.join().unwrap();
            }
        });
        assert_eq!(c, expected);

        // print the result
        for i in 0..a_row {
            for j in 0..b_row {
                //print!("{:?} ", c[i * b_row + j]);
            }
            //println!();
        }
    } */
}
