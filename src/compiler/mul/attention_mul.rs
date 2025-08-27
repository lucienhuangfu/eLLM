use std::f16;
use std::ops::{Add, Div, Mul, Sub};

use super::super::super::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::super::super::kernel;
use super::mul_trait::AttentionMulTrait;
use crate::compiler::assign::assign;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

#[derive(Clone)]
pub struct AttentionMul<T> {
    // chunks: Vec<(ConstPtr<T>, ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    q_ptr: ConstPtr<T>,
    k_ptr: ConstPtr<T>,
    v_ptr: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    batch_size: usize,
    attention_head_num: usize,
    kv_head_num: usize,
    head_size: usize,

    kv_strides: Vec<usize>,
    //v_strides: Vec<usize>,
    inverse_sqrt_head: T,
    // cpu_num: usize,
    // stride: usize,
}

impl<T> AttentionMul<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + NegInfinity
        + Exp,
{
    pub fn new(
        q_ptr: ConstPtr<T>,
        k_ptr: ConstPtr<T>,
        v_ptr: ConstPtr<T>,
        output_ptr: MutPtr<T>,
        batch_size: usize,
        head_size: usize,
        head_num: usize,
        kv_strides: Vec<usize>,
        // v_strides: Vec<usize>,
        inverse_sqrt_head: T,
        // stride: usize,
        // cpu_num: usize,
    ) -> Self {
        Self {
            q_ptr: q_ptr,
            k_ptr: k_ptr,
            v_ptr: v_ptr,
            output_ptr: output_ptr,
            batch_size: batch_size,
            head_num: head_num,
            head_size: head_size,
            kv_strides: kv_strides,
            inverse_sqrt_head: inverse_sqrt_head,
        }
    }

    // pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
    //     self.chunks = chunks;
    // }

    pub fn run(
        &self,
        position_index: usize,
        position_interval: usize,
        batch_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        // [sequence, batch, head_num, head_size]
        let mut q_ptr = self.q_ptr.ptr;
        let mut k_ptr = self
            .k_ptr
            .ptr
            .add(position_index * self.kv_strides[position_index]);
        let mut v_ptr = self
            .v_ptr
            .ptr
            .add(position_index * self.kv_strides[position_index]);
        let mut output_ptr = self.output_ptr.ptr;

        let stride = batch_size * self.attention_head_num;
        let max_stride = self.batch_size * self.attention_head_num;
        if let Some((begin, end)) = assign(position_interval * stride, cpu_num, thread_id) {
            // 从begin得到对应的坐标
            let (mut high_index, mut _index) = (begin / stride, begin % stride);
            let (mut row_index, mut col_index) = (
                _index / self.attention_head_num,
                _index % self.attention_head_num,
            );

            unsafe {
                // 遍历每个chunk;
                println!(
                    "thread_id: {}, begin: {}, end: {}",
                    thread_id,
                    begin,
                    end,
                    // self.chunks.len()
                );

                // [sequence, batch, head_num, head_size]
                for _ in begin..end {
                    let index = (high_index * max_stride + row_index * self.head_num + col_index)
                        * self.attention_head_size;

                    // println!(
                    //     " high_index: {}, row_index: {}, col_index: {}, index: {}",
                    //     high_index, row_index, col_index, index
                    // );
                    // Print values from self.ptr1 as slice
                    // let slice = std::slice::from_raw_parts(ptr1.add(index), self.head_size);
                    // println!("self.ptr1 slice at index {}: {:?}", index, slice);
                    // self.compute(ptr1.add(index), ptr2, output_ptr.add(index));
                    let _col_index = col_index / self.kv_head_num;

                    // [batch_size, head_num, sequence, head_size]
                    let kv_index = row_index * self.kv_strides[0]
                        + _col_index * self.kv_strides[1];

                    self.compute(
                        q_ptr.add(index),
                        k_ptr.add(kv_index),
                        v_ptr.add(kv_index),
                        output_ptr.add(index),
                        position_index + high_index,
                    );

                    col_index += 1;
                    if col_index == self.attention_head_num {
                        col_index = 0;
                        row_index += 1;
                    }
                    if row_index == batch_size {
                        row_index = 0;
                        high_index += 1;
                        // ptr2 = ptr2.add(self.head_size);
                    }
                }
            }
        }
    }
}

impl<T> AttentionMulTrait<T> for AttentionMul<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + NegInfinity
        + Exp,
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
            self.inverse_sqrt_head,
            self.head_size,
            self.kv_strides[2],
            position,
        );

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::flash_attention::flash_attention(
            input_ptr1,
            input_ptr2,
            input_ptr3,
            output_ptr,
            self.inverse_sqrt_head,
            self.head_size,
              self.kv_strides[2],
            position,
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
            self.kv_strides[2],
            position,
        );
    }
}

#[cfg(test)]
mod test {
    use super::super::chunk_attention::chunk_attention;
    use super::*;
    use crate::ptensor::tensor_utils::get_strides;
    use approx::assert_ulps_eq;

    #[test]
    fn test_attention_mul() {
        let sequence_chunk_size = 256;
        let sequence_length = 256;
        let batch_size = 16;
        let head_num = 8;
        let head_size = 128;
        
        let shape1 = vec![sequence_chunk_size, batch_size, head_num, head_size];
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

        let thread_num: usize = num_cpus::get();
        let mut operator =
            AttentionMul::<f32>::new(head_size, head_num, _strides2[2], 1.0, thread_num);
        operator.set_chunk(tasks);

        for i in 0..thread_num {
            // println!("{}", i);
            operator.run(batch_size, sequence_length - 1, i);
        }

        assert_ulps_eq!(data4[..], result[..], max_ulps = 4);
    }
}
