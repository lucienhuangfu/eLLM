// use num_traits::Float;
use std::f16;
use std::ops::{Add, Sub, Mul, Div, Neg};

use crate::kernel::generic::sigmoid::Sigmoid;
use super::zip_map_trait::ZipMapTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
// ::generic::complex_mul::complex_mul as complex_mul_block;
#[derive(Clone)]
pub struct ComplexZipMap<T> 
// where
//    T: Copy + Default + Add<Output = T>+ Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,
{
    chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    max_batch_size: usize,
    head_size: usize,
    head_num: usize,
    sequence_stride: usize,
    cpu_num: usize,
}

impl<T> ComplexZipMap<T> 
where
    T: Copy + Default + Add<Output = T>+ Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,

{
    pub fn new(
        max_batch_size: usize,
        head_size: usize,
        head_num: usize,
        batch_size: usize,
        cpu_num: usize,
    ) -> Self {
        Self {
            chunks: vec![],
            max_batch_size: max_batch_size,
            head_size: head_size,
            head_num: head_num,
            sequence_stride: batch_size * head_num,
            cpu_num: cpu_num,
        }
    }

    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }

    /* 
    //new version
    pub fn run(&self, batch_size: usize, position: usize, thread_id: usize) {
        let index = position * self.sequence_stride;
        let task_size = batch_size * self.head_num;
        // println!("head_size {}", self.head_size);
        // println!("sequence_stride {}", self.sequence_stride);
        // println!("task_size: {}, batch_size: {}, head_num: {}", task_size, batch_size, self.head_num);
        if let Some((begin, end)) = assign(task_size, self.cpu_num, thread_id) {
            // println!("chunk {}", self.chunks.len());
            // println!("p begin: {}, p end: {}, thread_id: {}", index+begin, index+end, thread_id);
            for (a, b, c) in self.chunks.get((index + begin)..(index + end)).unwrap() {
                self.compute(a.ptr, b.ptr, c.ptr);
            }
        }
    }*/

        pub fn run(&self,    
            batch_size: usize, 
            position_begin: usize,
            position_interval: usize,
            thread_id: usize) {
        let stride = batch_size * self.head_num;
        let max_stride = self.max_batch_size * self.head_num;
        if let Some((begin, end)) = assign(position_interval * stride, self.cpu_num, thread_id) {
      
            // 从begin得到对应的坐标
            let (mut high_index, mut _index) = (begin / stride, begin % stride);
            let (mut row_index, mut col_index) = (_index / batch_size, _index % batch_size);

            // 遍历每个chunk
            for i in begin..end {
                let index = high_index* max_stride + row_index * self.max_batch_size + col_index;
                unsafe {
                    let (a, b, c) = self.chunks.get_unchecked(index);
                    self.compute(a.ptr, b.ptr, c.ptr);
                }
                if col_index == self.head_num {
                    col_index = 0;
                    row_index += 1;
                }
                if row_index ==  batch_size {
                    row_index = 0;
                    high_index += 1;
                }
            }
        }
    }




}
// unsafe impl<T: Float> Send for ComplexZipMap<T> {}
// unsafe impl<T: Float> Sync for ComplexZipMap<T> {}

impl<T> ZipMapTrait<T> for ComplexZipMap<T> 
where
    T: Copy + Default + Add<Output = T>+ Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        //print!("generic runner\n");
        kernel::generic::complex_mul::complex_mul(
            input_ptr1 ,
            input_ptr2,
            output_ptr ,
            self.head_size,
        );
    }
}
impl ZipMapTrait<f16> for ComplexZipMap<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        //print!("f16 runner\n");
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::complex_mul::complex_mul(input_ptr1, input_ptr2, output_ptr, self.head_size);
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::complex_mul::complex_mul(input_ptr1, input_ptr2, output_ptr, self.head_size);
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
        kernel::generic::complex_mul::complex_mul(input_ptr1, input_ptr2 , output_ptr , self.head_size);
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
    use super::*;
    use super::super::chunk_zipmap::chunk_zipmap;
    use crate::ptensor::tensor_utils::{get_strides, get_broadcast_shape, get_aligned_strides};
    use approx::assert_ulps_eq;


    #[test]
    fn test_complexmul() {
        let head_size = 34;
        let head_num = 10;
        let batch_size = 10;
        let sequence_length = 10;
        let shapes = vec![sequence_length, batch_size, head_num, head_size];
        let length: usize = shapes.iter().product();
        let input_strides1 = get_strides(&shapes);
        let input_strides2 = input_strides1.clone();
        let output_strides = input_strides1.clone();
        let input_data1: Vec<f32> = (1..=34).cycle().take(34000).map(|x| x as f32).collect();
        let input_data2: Vec<f32> = (1..=34).cycle().take(34000).map(|x| x as f32).collect();
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
            ComplexZipMap::new(head_size, head_num, batch_size, thread_num); 
        _operator.set_chunk(chunks);

        for i in 0..thread_num {
            for sequence in 0..sequence_length {
                _operator.run(batch_size, sequence, i);
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
            ComplexZipMap::new(head_size, head_num, batch_size, thread_num);
        operator.set_chunk(chunks);

        for i in 0..thread_num {
            for position_index in 0..sequence_length {
                operator.run(1, position_index, i);
            }
        }

        assert_eq!(output_data[34..68], expected);
    }
}
