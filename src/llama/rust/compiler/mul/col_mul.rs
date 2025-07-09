use super::super::super::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::super::super::kernel;
use crate::compiler::assign::assign;
use std::ops::{Add, Mul};
use std::f16;


pub trait ColMulTrait<T> {
    fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
        position: usize,
    );
}

// unsafe impl Send for ColMulTrait {}
// unsafe impl Sync for ColMulTrait {}

#[derive(Clone)]
pub struct ColMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    head_size: usize,
    head_num: usize,
    cpu_num: usize,
}

impl<T> ColMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    pub fn new(
        //chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
        head_size: usize,
        head_num: usize,
        cpu_num: usize,
    ) -> Self {
        Self {
            chunks: vec![],
            head_size: head_size,
            head_num: head_num,
            cpu_num: cpu_num,
        }
    }

    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }

    pub fn run(&self, batch_size: usize, position: usize, thread_id: usize) {
        let (begin, end) = assign(batch_size * self.head_num, self.cpu_num, thread_id).unwrap();
        for (a, b, c) in self.chunks.get(begin..end).unwrap() {
            self.compute(a.ptr, b.ptr, c.ptr, position);
        }
    }
}

impl<T> ColMulTrait<T> for ColMul<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    default fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T,
        position: usize,
    ) {
        // kernel::generic::colmul::colmul(input_ptr1, input_ptr2, output_ptr, position , self.head_size);
    }
}

impl ColMulTrait<f32> for ColMul<f32> {
    fn compute(
        &self,
        input_ptr1: *const f32,
        input_ptr2: *const f32,
        output_ptr: *mut f32,
        position: usize,
    ) {
        kernel::generic::colmul::colmul(
            input_ptr1,
            input_ptr2,
            output_ptr,
            position,
            self.head_size,
        );
    }
}

#[cfg(test)]
mod test {
    use super::super::super::super::ptensor::{chunk_colmul::chunk_colmul, tensor_utils::get_strides};
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_col_mul() {
        let head_size = 128;
        let head_num = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let shape1 = vec![batch_size, head_num, sequence_length];
        let size1 = shape1.iter().product();
        let data1 = vec![1.0; size1];
        let strides1 = get_strides(&shape1);

        let shape2 = vec![batch_size, head_num, sequence_length, head_size];
        let size2 = shape2.iter().product();
        let data2 = vec![1.0; size2];
        let strides2 = get_strides(&shape2);

        let shape3 = vec![batch_size, head_num, head_size];
        let size3 = shape3.iter().product();
        let mut data3 = vec![0.0; size3];
        let strides3 = get_strides(&shape3);

        let result = vec![sequence_length as f32; size3];

        let chunks = chunk_colmul(
            data1.as_ptr(),
            shape1,
            strides1,
            data2.as_ptr(),
            shape2,
            strides2,
            data3.as_mut_ptr(),
            shape3,
            strides3,
        );

        let thread_num: usize = num_cpus::get();
        let mut operator = ColMul::<f32>::new(head_size, head_num, thread_num);
        operator.set_chunk(chunks);
        for i in 0..thread_num {
            operator.run(batch_size, sequence_length, i);
        }
        assert_ulps_eq!(data3[..], result[..], max_ulps = 4);
        // println!("{:?}", output);
    }
}
