use std::f16;
use std::ops::{Add, Sub, Mul, Div, Neg};

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::compiler::assign::assign;
use super::zip_map_trait::ZipMapTrait;
use crate::kernel;


#[derive(Clone)]
pub struct SiluZipMap<T> {
    chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    max_batch_size: usize,
    length: usize,
    h_num: usize,
    cpu_num: usize
}

impl <T> SiluZipMap <T> 
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T>+ Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,
{
    pub fn new(
        //chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
        max_batch_size: usize,
        length: usize, 
        h_num: usize, 
        cpu_num: usize
    ) -> Self { 
        Self {
            chunks: vec![],
            max_batch_size: max_batch_size,
            length: length,
            h_num: h_num,
            cpu_num: cpu_num
        }
    }

    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }

    pub fn run(&self,    
            batch_size: usize, 
            position_begin: usize,
            position_interval: usize,
            thread_id: usize) {
        let stride = batch_size * self.h_num;
        let max_stride = self.max_batch_size * self.h_num;
        if let Some((begin, end)) = assign(position_interval * stride, self.cpu_num, thread_id) {
      
            // 从begin得到对应的坐标
            let (mut high_index, mut _index) = (begin / stride, begin % stride);
            let (mut row_index, mut col_index) = (_index / self.h_num, _index % self.h_num);

            println!("thread_id: {}, begin: {}, end: {}, chunk num: {}", thread_id, begin, end, self.chunks.len());
            
            // 遍历每个chunk
            for i in begin..end {
                
                let index = high_index * max_stride + row_index * self.h_num + col_index;
                println!(" high_index: {}, row_index: {}, col_index: {}, index: {}",  high_index, row_index, col_index, index);
                unsafe {
                    let (a, b, c) = self.chunks.get_unchecked(index);
                    self.compute(a.ptr, b.ptr, c.ptr);
                }
                col_index += 1;
                if col_index == self.h_num {
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
// unsafe impl <T:Float>Send for SiluZipMap<T> {}
// unsafe impl <T:Float>Sync for SiluZipMap<T> {}

impl <T> ZipMapTrait  <T> for SiluZipMap<T>
where
    T: Copy + Default + Add<Output = T>+ Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2:*const T, output_ptr: *mut T, ) {
        // print!("generic runner\n");
        kernel::generic::silu::silu_multiply(input_ptr1 , input_ptr2, output_ptr , self.length);
    }
}
impl ZipMapTrait<f16> for SiluZipMap<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        // print!("f16 runner\n");
        
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::silu::silu_multiply(
                input_ptr1, input_ptr2 , output_ptr, self.length,
            );
        }; 
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::silu::silu_multiply(input_ptr1 , input_ptr2, output_ptr, self.length);      
    }
}

impl ZipMapTrait<f32> for SiluZipMap<f32> {
    fn compute (&self, input_ptr1: *const f32, input_ptr2: *const f32,output_ptr: *mut f32) {
        //print!("f32 runner\n");
        /*#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_silu_multiply_block(
                input_ptr1 , input_ptr2 , output_ptr , self.length
            );
        };
        // #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        kernel::generic::silu::silu_multiply(input_ptr1 , input_ptr2 , output_ptr , self.length);
    }
}
/* 
impl ZipMapTrait<f64> for SiluZipMap<f64> {
    fn compute (&self, input_ptr1: *const f64, input_ptr2: *const f64, output_ptr: *mut f64) {
        // print!("f64 runner\n");
        silu_multiply_block(input_ptr1 , input_ptr2 , output_ptr , self.length);   
    }
}*/

#[cfg(test)]
mod test {
    
    use super::*;
    use super::super::chunk_zipmap::chunk_zipmap;
    use crate::ptensor::tensor_utils::get_strides;
    use approx::assert_ulps_eq;
    use rand::seq;
    #[test]
    fn test_silu_multiply() {
        let sequence_threshold = 4;
        let batch_size = 32;
        let head_num = 64;
        let head_size = 128;
        // let hidden_size = 19;
        let shapes = vec![sequence_threshold, batch_size, head_num, head_size];
        let length = shapes.iter().product(); // 总元素数量


        let input_strides1 = get_strides(&shapes);
        //println!("{:?}", input_strides1);
        let input_strides2 = input_strides1.clone();
        let output_strides = input_strides1.clone();
        // let length: usize = shapes.iter().product();
        let input_data1: Vec<f32> = vec![
            2.1671206951141357,
            1.4490455389022827,
            -2.002431631088257,
            0.5662149786949158,
            0.3909946382045746,
            0.9437483549118042,
            -0.37030690908432007,
            0.7542704939842224,
            0.5875813961029053,
            1.6026240587234497,
            2.2485475540161133,
            -0.6622593402862549,
            -0.0015666020335629582,
            -0.5069465041160583,
            -0.37254711985588074,
            0.4420417249202728,
            -0.9305257201194763,
            0.5145581364631653,
            0.6260590553283691,
            2.1671206951141357,
            1.4490455389022827,
            -2.002431631088257,
            0.5662149786949158,
            0.3909946382045746,
            0.9437483549118042,
            -0.37030690908432007,
            0.7542704939842224,
            0.5875813961029053,
            1.6026240587234497,
            2.2485475540161133,
            -0.6622593402862549,
            -0.0015666020335629582,
            // -0.5069465041160583,
            // -0.37254711985588074,
            // 0.4420417249202728,
            // -0.9305257201194763,
            // 0.5145581364631653,
            // 0.6260590553283691,
        ].repeat(sequence_threshold*batch_size*head_num*4);
        let input_data2 = vec![1.0; length];
        let mut output_data: Vec<f32> = vec![0.0; length];

        let chunks = chunk_zipmap(shapes, input_data1.as_ptr(), input_strides1, input_data2.as_ptr(), input_strides2, output_data.as_mut_ptr(), output_strides);
        let thread_num: usize = num_cpus::get();
        let mut _operator: SiluZipMap<f32> = SiluZipMap::new(batch_size, head_size, head_num, thread_num);
        _operator.set_chunk(chunks);
        let position_index = 0; // 起始位置，根据实际情况可以修改
        for i in 0..thread_num {
            _operator.run(batch_size, position_index, sequence_threshold, i);
        }
        let result = vec![
            1.9444659948349,
            1.1735117435455322,
            -0.23818494379520416,
            0.36118248105049133,
            0.23323695361614227,
            0.6793630719184875,
            -0.15125809609889984,
            0.5129857659339905,
            0.3777032196521759,
            1.3339999914169312,
            2.033867835998535,
            -0.22532200813293457,
            -0.0007826874498277903,
            -0.1905660629272461,
            -0.15197153389453888,
            0.269090861082077,
            -0.2631694972515106,
            0.32204875349998474,
            0.4079371392726898,
            1.9444659948349,
            1.1735117435455322,
            -0.23818494379520416,
            0.36118248105049133,
            0.23323695361614227,
            0.6793630719184875,
            -0.15125809609889984,
            0.5129857659339905,
            0.3777032196521759,
            1.3339999914169312,
            2.033867835998535,
            -0.22532200813293457,
            -0.0007826874498277903,
        ].repeat(sequence_threshold*batch_size*head_num*4);
        //println!("{:?}",result.len());
        assert_ulps_eq!(output_data[..], result, max_ulps=4); 
        // println!("{:?}", output);
    }
}