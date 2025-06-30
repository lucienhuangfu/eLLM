use std::ops::{Add, Mul};
use super::super::super::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::super::super::kernel;
use crate::compiler::assign::assign;
use std::f16;
use kernel::generic::dot_product::dot_product as dot_product_block;
pub trait VecMulTrait<T> {
    fn compute(&self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        output_ptr: *mut T );
}

// unsafe impl Send for VecMulTrait {}
// unsafe impl Sync for VecMulTrait {}

#[derive(Clone)]
pub struct VecMul<T> 
where T: Copy + Add<Output = T> + Mul<Output = T> {
    chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    head_size: usize,
    head_num: usize,
    sequence_length: usize,
    cpu_num: usize
}

impl <T> VecMul <T> 
where T: Copy + Add<Output = T> + Mul<Output = T> {
    pub fn new(
            //chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
            head_size: usize,
            head_num: usize,
            sequence_length: usize,
            cpu_num:usize) -> Self { 
        Self {
            chunks: vec![],
            head_size: head_size,
            head_num: head_num,
            sequence_length: sequence_length,
            cpu_num:cpu_num
        }
    }

    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }

    pub fn run(&self,
            batch_size: usize,
            position: usize,
            thread_id: usize) {
        let mut index = 0;
        let (begin, end) = assign(position, self.cpu_num, thread_id).unwrap();
        for row_index in (0..(batch_size * self.head_num)) {
            for (a, b, c) in self.chunks.get((index+begin)..(index+end)).unwrap() {
                self.compute(a.ptr, b.ptr, c.ptr);
            }
            index += self.sequence_length;
        }
    }
}

impl <T> VecMulTrait<T> for VecMul<T> 
where T: Copy + Add<Output = T> + Mul<Output = T> {
    default fn compute(&self, 
            input_ptr1: *const T,
            input_ptr2: *const T,
            output_ptr: *mut T) {
        // kernel::dot_product(input_ptr1, input_ptr2, output_ptr, self.head_size);
    }
}

impl VecMulTrait<f16> for VecMul<f16> {
    fn compute(&self,  input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        /*#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
        SIMD_f16_512_vec_mul_block(
            input_ptr1, input_ptr2, output_ptr, self.head_size
        );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]*/
        dot_product_block(input_ptr1, input_ptr2, output_ptr, self.head_size);
    }
}
impl VecMulTrait<f32> for VecMul<f32> {
    fn compute(&self, 
            input_ptr1: *const f32,
            input_ptr2: *const f32,
            output_ptr: *mut f32) {
            /*#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
            SIMD_f32_256_vec_mul_block(
                input_ptr1, input_ptr2, output_ptr, self.head_size
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        dot_product_block(input_ptr1, input_ptr2, output_ptr, self.head_size);
    }
}
impl VecMulTrait<f64> for VecMul<f64> {
    fn compute(&self, 
            input_ptr1: *const f64,
            input_ptr2: *const f64,
            output_ptr: *mut f64) {
        dot_product_block(input_ptr1, input_ptr2, output_ptr, self.head_size);
    }
}
#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use super::*;
    // use super::super::super::super::ptensor::stride::get_strides;
    use super::super::super::super::ptensor::{chunk_vecmul::chunk_vecmul,tensor_utils::get_strides};
    use super::super::super::super::kernel;

    use std::slice;

    #[test]
    fn test_chunk_vec() {
        
        let head_size = 128;
        let head_num  = 64;
        let batch_size = 16;
        let sequence_length = 256;

        let q_shape = vec![batch_size, head_num, 1, head_size];
        let q_size = q_shape.iter().product();
        let q_data: Vec<f32> = vec![1.0; q_size];
        let q_strides = get_strides(&q_shape);


        let k_shape = vec![batch_size, head_num, sequence_length,  head_size];
        let k_size = k_shape.iter().product();
        let k_data: Vec<f32> = vec![1.0; k_size];
        let k_strides = get_strides(&k_shape);

        let s_shape = vec![batch_size , head_num, 1, sequence_length];
        let s_size = s_shape.iter().product();
        let mut s_data: Vec<f32> = vec![0.0 ; s_size];
        let s_strides = get_strides(&s_shape);

        let result = vec![head_size as f32; s_size];

        let chunks = chunk_vecmul(
            q_data.as_ptr(),
            q_shape,
            q_strides,
            k_data.as_ptr(),
            k_shape,
            k_strides,
            s_data.as_mut_ptr(),
            s_shape,
            s_strides);

        let thread_num: usize = num_cpus::get();
        let mut operator = VecMul::<f32>::new( head_size, head_num, sequence_length,thread_num);
        operator.set_chunk(chunks);
        for i in 0..thread_num {
            operator.run(batch_size, sequence_length,i);
        }

        assert_ulps_eq!(s_data[..], result[..], max_ulps=4);
    }

}

