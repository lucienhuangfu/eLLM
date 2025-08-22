// use num_traits::Float;
use std::f16;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::zip_map_trait::ZipMapTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::kernel::generic::sigmoid::Sigmoid;
// ::generic::complex_mul::complex_mul as complex_mul_block;
#[derive(Clone)]
pub struct ComplexZipMap<T> {
    // chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    ptr1: *const T,
    ptr2: *const T,
    output_ptr: *mut T,
    batch_size: usize,
    head_size: usize,
    head_num: usize,
    sequence_stride: usize,
    cpu_num: usize,
}

impl<T> ComplexZipMap<T>
where
    T: Copy
        + Default
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sigmoid<T>,
{
    pub fn new(
        ptr1: *const T,
        ptr2: *const T,
        output_ptr: *mut T,
        batch_size: usize,
        head_size: usize,
        head_num: usize,
        cpu_num: usize,
    ) -> Self {
        Self {
            ptr1: ptr1,
            ptr2: ptr2,
            output_ptr: output_ptr,
            // chunks: vec![],
            batch_size: batch_size,
            head_size: head_size,
            head_num: head_num,
            sequence_stride: batch_size * head_num,
            cpu_num: cpu_num,
        }
    }

    /* 
    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }*/

    pub fn run(
        &self,
        batch_size: usize,
        position_begin: usize,
        position_interval: usize,
        thread_id: usize,
    ) {
        let stride = batch_size * self.head_num;
        let max_stride = self.batch_size * self.head_num;
        if let Some((begin, end)) = assign(position_interval * stride, self.cpu_num, thread_id) {
            // 从begin得到对应的坐标
            let (mut high_index, mut _index) = (begin / stride, begin % stride);
            let (mut row_index, mut col_index) = (_index / self.head_num, _index % self.head_num);

            unsafe {
                let mut _ptr2 = self.ptr2.add(position_begin * self.head_size);
                // 遍历每个chunk;

                println!(
                    "thread_id: {}, begin: {}, end: {}",
                    thread_id,
                    begin,
                    end,
                    // self.chunks.len()
                );
                for i in begin..end {
                    let index = (high_index * max_stride + row_index * self.head_num + col_index)* self.head_size;
                    println!(
                        " high_index: {}, row_index: {}, col_index: {}, index: {}",
                        high_index, row_index, col_index, index
                    );

                    // Print values from self.ptr1 as slice
               
                    let slice = std::slice::from_raw_parts(self.ptr1.add(index), self.head_size);
                    println!("self.ptr1 slice at index {}: {:?}", index, slice);

                    self.compute(self.ptr1.add(index), _ptr2, self.output_ptr.add(index));

                    col_index += 1;
                    if col_index == self.head_num {
                        col_index = 0;
                        row_index += 1;
                    }
                    if row_index == batch_size {
                        row_index = 0;
                        high_index += 1;
                        _ptr2 = _ptr2.add(self.head_size);
                    }
                }
            }
        }
    }
}
// unsafe impl<T: Float> Send for ComplexZipMap<T> {}
// unsafe impl<T: Float> Sync for ComplexZipMap<T> {}

impl<T> ZipMapTrait<T> for ComplexZipMap<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sigmoid<T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        //print!("generic runner\n");
        kernel::generic::complex_mul::complex_mul(
            input_ptr1,
            input_ptr2,
            output_ptr,
            self.head_size,
        );
    }
}
impl ZipMapTrait<f16> for ComplexZipMap<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        //print!("f16 runner\n");
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::complex_mul::complex_mul(
                input_ptr1,
                input_ptr2,
                output_ptr,
                self.head_size,
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::complex_mul::complex_mul(
            input_ptr1,
            input_ptr2,
            output_ptr,
            self.head_size,
        );
    }
}

impl ZipMapTrait<f32> for ComplexZipMap<f32> {
    fn compute(&self, input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32) {
        //print!("f32 runner\n");
        /*#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_complex_mul_block(
                input_ptr1 , input_ptr2 , output_ptr, self.head_size
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        kernel::generic::complex_mul::complex_mul(
            input_ptr1,
            input_ptr2,
            output_ptr,
            self.head_size,
        );
    }
}
impl ZipMapTrait<f64> for ComplexZipMap<f64> {
    fn compute(&self, input_ptr1: *const f64, input_ptr2: *const f64, output_ptr: *mut f64) {
        //print!("f64 runner\n");
        // complex_mul_block(input_ptr1 , input_ptr2 , output_ptr , self.head_size);
    }
}

#[cfg(test)]
mod test {
    use super::super::chunk_zipmap::chunk_zipmap;
    use super::*;
    use crate::ptensor::tensor_utils::{get_aligned_strides, get_broadcast_shape, get_strides};
    use approx::assert_ulps_eq;
    // use nom::sequence;
    // use rand::seq;

    #[test]
    fn test_complexmul() {
        let head_size = 34;
        let input_data1: Vec<f32> = (1..=34).cycle().take(head_size).map(|x| x as f32).collect();
        let input_data2: Vec<f32> = (1..=34).cycle().take(head_size).map(|x| x as f32).collect();
        let mut output_data: Vec<f32> = vec![0.0; head_size];

        let input_ptr1 = input_data1.as_ptr();
        let input_ptr2 = input_data2.as_ptr();
        let output_ptr = output_data.as_mut_ptr();

        kernel::generic::complex_mul::complex_mul(input_ptr1, input_ptr2, output_ptr, head_size);

        let expected: Vec<f32> = vec![
            -3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0,
            364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0,
            1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0,
        ];

        assert_eq!(output_data, expected);
    }

    #[test]
    fn test_complexmul2() {
        let sequence_length = 10;
        let sequence_threshold = 4;
        let batch_size = 10;
        let head_num = 10;
        let head_size = 34;

        let shape1 = vec![sequence_threshold, batch_size, head_num, head_size];
        let shape2 = vec![sequence_length, head_size];

        // let broadcast_shape = get_broadcast_shape(&shape1, &shape2);

        // let input_strides1 = get_aligned_strides(&shape1, &broadcast_shape);
        // let input_strides2 = get_aligned_strides(&shape2, &broadcast_shape);
        // let output_strides = get_strides(&broadcast_shape);

        let length1: usize = shape1.iter().product();
        let length2: usize = shape2.iter().product();
        let length: usize = shape1.iter().product();
        let input_data1: Vec<f32> = (1..=34).cycle().take(length1).map(|x| x as f32).collect();
        let input_data2: Vec<f32> = (1..=34).cycle().take(length2).map(|x| x as f32).collect();
        let mut output_data: Vec<f32> = vec![0.0; length];

        let expected: Vec<f32> = vec![
            -3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0,
            364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0,
            1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0,
        ];

        /*
        let chunks = chunk_zipmap(
            broadcast_shape,
            input_data1.as_ptr(),
            input_strides1,
            input_data2.as_ptr(),
            input_strides2,
            output_data.as_mut_ptr(),
            output_strides,
        ); */

        let thread_num: usize = num_cpus::get();
        let mut operator: ComplexZipMap<f32> = ComplexZipMap::new(
            input_data1.as_ptr(),
            input_data2.as_ptr(),
            output_data.as_mut_ptr(),
            batch_size,
            head_size,
            head_num,
            thread_num,
        );
        // operator.set_chunk(chunks);
        let position_index = 0; // Assuming we want to run for the first position

        for i in 0..thread_num {
            // for position_index in 0..sequence_length {
            operator.run(batch_size, position_index, sequence_threshold, i);
            // }
            break;
        }

        assert_eq!(output_data[34..68], expected);
    }

    /*
        #[test]
        fn test_complexmul() {
            let sequence_length = 10;
            let batch_size = 10;
            let head_num = 10;
            let head_size = 34;

            let shapes = vec![sequence_length, batch_size, head_num, head_size];
            let length: usize = shapes.iter().product();
            let input_strides1 = get_strides(&shapes);
            let input_strides2 = input_strides1.clone();
            let output_strides = input_strides1.clone();
            let input_data1: Vec<f32> = (1..=head_size).cycle().take(34000).map(|x| x as f32).collect();
            let input_data2: Vec<f32> = (1..=head_size).cycle().take(34000).map(|x| x as f32).collect();
            let mut output_data: Vec<f32> = vec![0.0; length];
            // let input_data1: Vec<f16> = input_data1.into_iter().map(|x| x)).collect();
            // let input_data2: Vec<f16> = input_data2.into_iter().map(|x| x)).collect();
            let expected: Vec<f32> = vec![
                -3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0,
                364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0,
                1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0,
            ];
            // let expected: Vec<f32> = expected.into_iter().map(|x| x)).collect();
            let chunks = chunk_zipmap(
                shapes,
                input_data1.as_ptr(),
                input_strides1,
                input_data2.as_ptr(),
                input_strides2,
                output_data.as_mut_ptr(),
                output_strides,
            );
            let thread_num: usize = num_cpus::get();
            let mut _operator: ComplexZipMap<f32> =
                ComplexZipMap::new(batch_size, head_size, head_num, batch_size, thread_num);
            _operator.set_chunk(chunks);

            for i in 0..thread_num {
                for sequence in 0..sequence_length {
                    _operator.run(batch_size, 0, sequence, i);
                }
            }
            //println!("{:?}",output_data[34]);
            assert_eq!(output_data[34..68], expected);
            // println!("{:?}", output);
        }




        #[test]
        fn test_complexmul_with_broadcast() {
            let head_size = 34;
            let head_num = 10;
            let batch_size = 10;
            let sequence_length = 10;

            let shape1 = vec![batch_size, head_num, head_size];
            let shape2 = vec![sequence_length, 1, 1, head_size];
            let broadcast_shape = get_broadcast_shape(&shape1, &shape2);

            let length: usize = broadcast_shape.iter().product();
            let input_strides1 = get_aligned_strides(&shape1, &broadcast_shape);
            let input_strides2 = get_aligned_strides(&shape2, &broadcast_shape);
            let output_strides = get_strides(&broadcast_shape);

            let length1: usize = shape1.iter().product();
            let length2: usize = shape2.iter().product();
            let input_data1: Vec<f32> = (1..=34).cycle().take(length1).map(|x| x as f32).collect();
            let input_data2: Vec<f32> = (1..=34).cycle().take(length2).map(|x| x as f32).collect();
            let mut output_data: Vec<f32> = vec![0.0; length];

            let expected: Vec<f32> = vec![
                -3.0, 4.0, -7.0, 24.0, -11.0, 60.0, -15.0, 112.0, -19.0, 180.0, -23.0, 264.0, -27.0,
                364.0, -31.0, 480.0, -35.0, 612.0, -39.0, 760.0, -43.0, 924.0, -47.0, 1104.0, -51.0,
                1300.0, -55.0, 1512.0, -59.0, 1740.0, -63.0, 1984.0, -67.0, 2244.0,
            ];

            let chunks = chunk_zipmap(
                broadcast_shape,
                input_data1.as_ptr(),
                input_strides1,
                input_data2.as_ptr(),
                input_strides2,
                output_data.as_mut_ptr(),
                output_strides,
            );

            let thread_num: usize = num_cpus::get();
            let mut operator: ComplexZipMap<f32> =
                ComplexZipMap::new(batch_size, head_size, head_num, batch_size, thread_num);
            operator.set_chunk(chunks);

            for i in 0..thread_num {
                for position_index in 0..sequence_length {
                    operator.run(1, position_index, i);
                }
            }

            assert_eq!(output_data[34..68], expected);
        }
    */
}
