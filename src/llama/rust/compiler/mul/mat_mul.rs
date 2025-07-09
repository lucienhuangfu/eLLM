
use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::{Arc, Barrier};
use std::f16;

use super::mul_trait::MatlMulTrait;
use super::super::super::init::{send_sync_ptr::{ConstPtr, MutPtr}, matmul_params::MatMulParams};
use super::super::super::kernel;
use super::super::assign::assign;



// there will be just one instance of this runner in the program
// this runner will be shared by many threads that together compute the matrix multiplication
#[derive(Clone)]
pub struct MatMul<T>
{
    pub params: MatMulParams,
    _marker: PhantomData<T>,

    // all tasks for all threads, need to find the tasks of the current thread to run
    chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    sequence_length: usize,
    sequence_stride: usize,
    cpu_num: usize,
    // barrier for synchronization, need to wait at the finishing a row of macro kernels of matrix B
    barrier_arc: Arc<Barrier>,
}
impl<T> MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn new(
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
        // this tasks vector is owned by this runner after being returned by the chunk_matmul function
        //chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
        sequence_length: usize,
        cpu_num: usize,
        // this is the initial Arc wrapper of the barrier object
        barrier_arc: Arc<Barrier>,
    ) -> Self {
        Self {
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
            chunks: vec![],
            sequence_length: sequence_length,
            sequence_stride: a_row * b_row,
            cpu_num: cpu_num,
            barrier_arc: barrier_arc,
        }
    }

     // this run method is for a single thread to take and run tasks from the queue
    // batch size is the length of user input, and it is the number of rows in the left matrix
    // this runner, essentially the parameters in it, will be shared by many threads that together compute the matrix multiplication
    // the thread_num is the number of threads that will be used in the computation
    // the thread_id is the id of the current thread, in range [0, thread_num)
    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }
    pub fn run(&self, batch_size: usize, position_index: usize, thread_id: usize) {


        // special case when the batch size is 1
        if batch_size == 1 {
            // the left matrix is just a vector
            // the right matrix is a matrix
            // just let the left vector do dot product with each row of the right matrix
            // each thread gets a continuous chunk of rows of the right matrix
            let (begin, end) = assign(self.params.b_row, self.cpu_num, thread_id).unwrap();
            // a_ptr is the pointer to the leftmost, or first element of the left vector
            // b_ptr is the pointer to the cell at the first row and first column of right matrix
            // c_ptr is the pointer to the cell at the first row and first column of the output matrix 
            let (a_ptr, b_ptr, mut c_ptr) = self.chunks[0];
            if self.sequence_length != 1 {
                unsafe {
                    c_ptr.ptr = c_ptr.ptr.add(position_index * self.sequence_stride);
                }
            } 
            // i is the index of a row of the right matrix

            // println!("{} {} {} {}",thread_id, begin, end, self.params.column);
            for i in (begin..end) {
                // calculate a cell in c as the dot product of the left vector and a row of the right matrix
                unsafe {
                    self.compute2(
                        a_ptr.ptr,
                        b_ptr.ptr.add(i * self.params.column),
                        c_ptr.ptr.add(i),
                        self.params.column,
                    );
                }
            }
            return;
        }

        // Todo: 需要实现append tail of kv tensor
        let barrier_clone = Arc::clone(&(self.barrier_arc));
        let num_tasks = self.chunks.len();
        // this is a chunk that a barrier will be placed after the completion of the chunk
        let chunk_size = self.params.b_row / self.params.b_row_step_macro;
        // use assign to decide the parce of tasks inside a chunk
        let (begin, end) = assign(chunk_size, self.cpu_num, thread_id).unwrap();
        // print thread_id, begin, end, chunk_size
       /*  println!(
            "thread_id: {}, begin: {}, end: {}, chunk_size: {}",
            thread_id, begin, end, chunk_size
        );*/

        // this is how many macro kernels are there in the column direction
        let column_macro_num = self.params.column / self.params.column_step_macro;
        // a_row_macro_full_num is how many macro kernels are there in the row direction
        // a_row_remainder is the number of rows that are not enough to form a full macro kernel, it is the height of the partial macro kernel
        let (a_row_macro_full_num, a_row_remainder) = (
            batch_size / self.params.a_row_step_macro,
            batch_size % self.params.a_row_step_macro,
        );

        // for the first full_macro_num chunks, all macro kernels are full and can be run without additional care
        let full_macro_num = a_row_macro_full_num * column_macro_num;
        // for the last half_macro_num chunks, the left matrix is smaller than one full macro kernel
        let partial_macro_num = if a_row_remainder == 0 {
            0
        } else {
            column_macro_num
        };
        // print there are full_macro_num full macro kernels and partial_macro_num partial macro kernels
        /* 
        println!(
            "thread_id: {} full_macro_num: {}, partial_macro_num: {}",
            thread_id, full_macro_num, partial_macro_num
        );*/
        // process the full tasks
        // upperbound marks a position in chunks, the list of tasks, that is a checkpoint for all threads to reach to advance to the next chunk
        // i stands for the ith macro kernel of matrix A, it is 1 indexed
        // 1 stands for the first macro kernel of matrix A
        // 2 stands for the second macro kernel of matrix A, sharing the same row with the first macro kernel of matrix A

        for i in (1..full_macro_num + 1) {
            let upperbound = i * chunk_size;
            for j in (begin..end) {
                let task: (ConstPtr<T>, ConstPtr<T>, MutPtr<T>) =
                    self.chunks[upperbound - chunk_size + j];
                let a_ptr = unsafe { task.0.ptr };
                let b_ptr = unsafe { task.1.ptr };
                let c_ptr = unsafe { task.2.ptr };
                for x in (0..self.params.a_row_step_macro).step_by(self.params.a_row_step_micro) {
                    for y in (0..self.params.b_row_step_macro).step_by(self.params.b_row_step_micro)
                    {
                        //println!("thread_id: {} micro kernel ", thread_id);

                        unsafe {
                            self.compute(
                                a_ptr.add(x * self.params.column),
                                b_ptr.add(y * self.params.column),
                                c_ptr.add(x * self.params.b_row + y),
                            );
                        }
                    }
                }
            }
            // print thread id wait at barrier upperbound
            //println!("thread_id: {} wait at barrier {}", thread_id, upperbound);
            barrier_clone.wait();
        }
        // process the partial tasks
        // i stands for the ith incomplete macro kernel of matrix A, it is 1 indexed
        // 1 stands for the first incomplete macro kernel of matrix A
        // 2 stands for the second incomplete macro kernel of matrix A, sharing the same row with the first incomplete macro kernel of matrix A
        for i in (1..partial_macro_num + 1) {
            let upperbound = full_macro_num * chunk_size + i * chunk_size;
            for i in (upperbound - chunk_size + begin..upperbound - chunk_size + end) {
                let task: (ConstPtr<T>, ConstPtr<T>, MutPtr<T>) = self.chunks[i];
                let a_ptr = unsafe { task.0.ptr };
                let b_ptr = unsafe { task.1.ptr };
                let c_ptr = unsafe { task.2.ptr };
                // a_row_remainder is the height of the partial macro kernel
                for x in (0..a_row_remainder).step_by(self.params.a_row_step_micro) {
                    for y in (0..self.params.b_row_step_macro).step_by(self.params.b_row_step_micro)
                    {
                        //println!("thread_id: {} micro kernel ", thread_id);
                        unsafe {
                            self.compute(
                                a_ptr.add(x * self.params.column),
                                b_ptr.add(y * self.params.column),
                                c_ptr.add(x * self.params.b_row + y),
                            );
                        }
                    }
                }
            }
            // print thread id wait at barrier upperbound
            //println!("thread_id: {} wait at barrier {}", thread_id, upperbound);
            barrier_clone.wait();
        }
        // warn: it is not tested what will happen when the micro kernel access rows larger than the batch_size of the matrix
    }
}


impl<T> MatlMulTrait<T> for MatMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        //print!("generic runner\n");
        kernel::generic::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &(self.params));
    }

    default fn compute2(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T, length: usize) {
        
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }

}

impl MatlMulTrait<f16> for MatMul<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        // print!("f16 runner\n");
         
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::matmul_block::  matmul_block(
                 input_ptr1, input_ptr2, output_ptr, &self.params);
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::matmul_block::matmul_block(input_ptr1, input_ptr2, output_ptr, &(self.params));
    }

    fn compute2(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16, length: usize) {
        // print!("f16 runner\n");
        
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);

        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::dot_product::dot_product(input_ptr1, input_ptr2, output_ptr, length);
    }

}

impl MatlMulTrait<f32> for MatMul<f32> {
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

    fn compute2(&self, input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32, length: usize) {
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
    use std::thread;

    use super::super::chunk_matmul::chunk_matmul;

    #[test]
    fn test_f32_single() {
        let batch_size = 1;
        let a_row = 128;
        let b_row = 256;
        let column = 256;
        let a_row_step_macro = 32;
        let b_row_step_macro = 32;
        let column_step_macro = 64;
        let a_row_step_micro = 8;
        let b_row_step_micro = 8;

        let thread_num: usize = num_cpus::get();

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
    }

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

    // test f32 runner
    #[test]
    fn test_f32_runner_sequence() {
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
        let mut expected = vec![0.0; sequence_length *a_row * b_row];
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

}
