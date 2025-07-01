use std::ops::{Add, Sub, Mul, Div};
// use num_traits::Float;
use std::f16;

use super::mul_trait::AttentionMulTrait;
use super::super::super::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::super::super::kernel;
use crate::compiler::assign::assign;
use crate::kernel::generic::{neg_infinity::NegInfinity, exp::Exp};



#[derive(Clone)]
pub struct AttentionMul<T>
// where T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialOrd + NegInfinity + Exp,
{
    chunks: Vec<(ConstPtr<T>, ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    head_size: usize,
    head_num: usize,
    stride: usize,
    cpu_num: usize,
    inverse_sqrt_head: T,
}

impl<T> AttentionMul<T>
where T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialOrd + NegInfinity + Exp,
{
    pub fn new(
        head_size: usize,
        head_num: usize,
        stride: usize,
        inverse_sqrt_head: T,
        cpu_num: usize,
    ) -> Self {
        Self {
            chunks: vec![],
            head_size: head_size,
            inverse_sqrt_head: inverse_sqrt_head,
            head_num: head_num,
            stride: stride,
            cpu_num: cpu_num,
        }
    }

    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }

    pub fn run(&self, batch_size: usize, position: usize, thread_id: usize) {
        if let Some((begin, end)) = assign(batch_size * self.head_num, self.cpu_num, thread_id) {
            for (a, b, c, d) in self.chunks.get(begin..end).unwrap() {
                self.compute(a.ptr, b.ptr, c.ptr, d.ptr, position);
            }
        }

    }
}

impl<T> AttentionMulTrait<T> for AttentionMul<T>
where T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialOrd + NegInfinity + Exp,
{
    default fn compute(
        &self,
        input_ptr1: *const T,
        input_ptr2: *const T,
        input_ptr3: *const T,
        output_ptr: *mut T,
        position: usize,
    ) {
        kernel::generic::flash_attention::flash_attention(
            input_ptr1,
            input_ptr2,
            input_ptr3,
            output_ptr,
            self.inverse_sqrt_head,
            self.head_size,
            self.stride,
            position,
        );
    }
}

impl AttentionMulTrait<f16> for AttentionMul<f16> {
    fn compute(
        &self,
        input_ptr1: *const f16,
        input_ptr2: *const f16,
        input_ptr3: *const f16,
        output_ptr: *mut f16,
        position: usize,
    ) {
        
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::flash_attention::flash_attention(
            input_ptr1,
            input_ptr2,
            input_ptr3,
            output_ptr,
            self.head_size,
            self.stride,
            position
        );
        
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::flash_attention::flash_attention(
            input_ptr1,
            input_ptr2,
            input_ptr3,
            output_ptr,
            self.inverse_sqrt_head,
            self.head_size,
            self.stride,
            position
        );
    }
}

impl AttentionMulTrait<f32> for AttentionMul<f32> {
    fn compute(
        &self,
        input_ptr1: *const f32,
        input_ptr2: *const f32,
        input_ptr3: *const f32,
        output_ptr: *mut f32,
        position: usize,
    ) {
        kernel::generic::flash_attention::flash_attention(
            input_ptr1,
            input_ptr2,
            input_ptr3,
            output_ptr,
            self.inverse_sqrt_head,
            self.head_size,
            self.stride,
            position
        );
    }
}




#[cfg(test)]
mod test {
    use super::super::chunk_attention::chunk_attention;
    use crate::ptensor::tensor_utils::get_strides;
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_attention_mul() {
        let head_size = 128;
        let head_num = 8;
        let batch_size = 16;
        let sequence_length = 256;

        let shape1 = vec![batch_size, head_num, head_size];
        let size1 = shape1.iter().product();
        let data1 = vec![1.0; size1];
        let strides1 = get_strides(&shape1);

        let shape2 = vec![sequence_length, batch_size, head_num, head_size];
        let strides2 = get_strides(&shape2);
        let size2 = shape2.iter().product();
        let data2 = vec![1.0; size2];

        let _shape2 = vec![batch_size, head_num, sequence_length, head_size];
        let _strides2 = vec![strides2[1], strides2[2], strides2[0], strides2[3]];

        let shape3 = vec![batch_size, sequence_length, head_num, head_size];
        let size3 = shape3.iter().product();
        let data3 = vec![1.0; size3];

        let shape4 = vec![batch_size, head_num, head_size];
        let size4 = shape4.iter().product();
        let mut data4 = vec![0.0; size4];
        let strides4 = get_strides(&shape4);

        let result = vec![1.0; size4];

        let tasks = chunk_attention(
            data1.as_ptr(),
            shape1,
            strides1,
            data2.as_ptr(),
            _shape2,
            _strides2.clone(),
            data3.as_ptr(),
            data4.as_mut_ptr(),
            shape4,
            strides4,
        );

        let thread_num: usize = num_cpus::get();
        let mut operator = AttentionMul::<f32>::new(head_size, head_num, _strides2[2], 1.0, thread_num);
        operator.set_chunk(tasks);

        for i in 0..thread_num {
            // println!("{}", i);
            operator.run(batch_size, sequence_length - 1, i);
        }

        assert_ulps_eq!(data4[..], result[..], max_ulps = 4);
    }
}
