use std::f16;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use super::super::super::init::{
    matmul_params::MatmulParams,
    send_sync_ptr::{ConstPtr, MutPtr},
};
use super::super::super::kernel;
use super::super::assign::assign;
use super::mul_trait::MatmulTopKTrait;

// there will be just one instance of this runner in the program
// this runner will be shared by many threads that together compute the matrix multiplication
#[derive(Clone)]
pub struct MatmulTopK<T> {
    ptr1: ConstPtr<T>,
    ptr2: ConstPtr<T>,
    indice_ptr: MutPtr<usize>,
    value_ptr: MutPtr<T>,
    sum_ptr: MutPtr<T>,
    a_row: usize,
    b_row: usize,
    column: usize,
    pub params: MatmulParams,
    _marker: PhantomData<T>,
}
impl<T> MatmulTopK<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn new(
        ptr1: *const T,
        ptr2: *const T,
        indice_ptr: *mut usize,
        value_ptr: *mut T,
        sum_ptr: *mut T,
        a_row: usize,
        b_row: usize,
        column: usize,
        a_row_step_macro: usize,
        b_row_step_macro: usize,
        column_step_macro: usize,
        a_row_step_micro: usize,
        b_row_step_micro: usize,
    ) -> Self {
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: ptr2 },
            indice_ptr: MutPtr { ptr: indice_ptr },
            value_ptr: MutPtr { ptr: value_ptr },
            sum_ptr: MutPtr { ptr: sum_ptr },
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
            _marker: PhantomData,
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
        // 任务数远大于核数，每个核分配多个任务，但是只写一份结果
        //  c小块的列数最小16，写满avx512寄存器,, 只存8个
        // output [sequence_chunk_size, batch_size, thread_num, topk_size]
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

impl<T> MatmulTopKTrait<T> for MatmulTopK<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        indice_ptr: *mut usize,
        value_ptr: *mut T,
        sum_ptr: *mut T,
    ) {
        /*
        //print!("generic runner\n");
        kernel::generic::matmul_block::matmul_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            &(self.params),
        );*/
    }
}

impl MatmulTopKTrait<f16> for MatmulTopK<f16> {
    fn compute(
        &self,
        input_ptr1: *const f16,
        input_ptr2: *const f16,
        indice_ptr: *mut usize,
        value_ptr: *mut f16,
        sum_ptr: *mut f16,
    ) {
        /*
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
        ); */
    }
}

impl MatmulTopKTrait<f32> for MatmulTopK<f32> {
    fn compute(
        &self,
        input_ptr1: *const f32,
        input_ptr2: *const f32,
        indice_ptr: *mut usize,
        value_ptr: *mut f32,
        sum_ptr: *mut f32,
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
    } */

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

        // get the tasks
        let chunks = chunk_matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &params);
        // initialize the runner and multi-threaded run
        // create as many threads as the logical cores in the machine
        let thread_num: usize = num_cpus::get();
        let barrier = Barrier::new(thread_num);
        let barrier_arc = Arc::new(barrier);
        let mut runner = matmul::<f32>::new(
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
        let mut runner = matmul::<f16>::new(
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
        let mut runner = matmul::<f32>::new(
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
