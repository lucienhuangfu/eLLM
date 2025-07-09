
use std::f16;
use std::ops::{Add,Sub, Mul, Div, Neg};

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::zip_map_trait::ZipMapTrait;
use crate::compiler::assign::assign;
use crate::kernel;


#[derive(Clone)]
pub struct AddZipMap <T> {
    chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    length: usize,
    h_num: usize, 
    cpu_num: usize
}

impl <T> AddZipMap <T> 
where T: Copy + Default + Add<Output = T> + Sub<Output = T>+ Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,
{
    pub fn new(
        length: usize,
        h_num: usize,
        cpu_num:usize
    ) -> Self { 
        Self {
            chunks: vec![],
            length: length,
            h_num: h_num,
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
        if let Some((begin, end)) = assign(batch_size*self.h_num, self.cpu_num, thread_id) {
            // let begin = 0;
            // let end = batch_size;
            for &(a, b, c) in self.chunks.get(begin..end).unwrap() {
                self.compute(a.ptr, b.ptr, c.ptr);
            }
        }

    }
}

// unsafe impl <T:Float>Send for AddZipMap<T> {}
// unsafe impl <T:Float>Sync for AddZipMap<T> {}
impl <T> ZipMapTrait <T> for AddZipMap  <T>
where T: Copy + Default + Add<Output = T>+ Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,
{
    
    default fn compute(&self, input_ptr1: *const T, input_ptr2:*const T, output_ptr: *mut T, ) {
        // print!("generic \n");
        kernel::generic::add::add(input_ptr1 , input_ptr2 , output_ptr , self.length);
    }
}

impl ZipMapTrait<f16> for AddZipMap<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        // print!("f16 \n");
        
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::add::add(
                input_ptr1, input_ptr2, output_ptr, self.length,
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::add::add(input_ptr1 , input_ptr2, output_ptr , self.length);      
    }
}

impl ZipMapTrait<f32> for AddZipMap<f32> {
    fn compute (&self, input_ptr1: *const f32, input_ptr2: *const f32,output_ptr: *mut f32) {
        // print!("f32 \n");
        /*#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_add_block(
                input_ptr1, input_ptr2 , output_ptr , self.length
            );
        };
        // #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        kernel::generic::add::add(input_ptr1, input_ptr2, output_ptr , self.length);
    }
}
impl ZipMapTrait<f64> for AddZipMap<f64> {
    fn compute (&self, input_ptr1: *const f64, input_ptr2: *const f64, output_ptr: *mut f64) {
        // print!("f64 \n");
        // add_block(input_ptr1 , input_ptr2 , output_ptr , self.length);   
    }
}


#[cfg(test)]
mod test {
    

    use super::*;
    use super::super::chunk_zipmap::chunk_zipmap;
    use crate::ptensor::tensor_utils::get_strides;
    use approx::assert_ulps_eq;

    #[test]
    fn test_add_zip() {
        let batch_size = 2; // 每次批处理 10 个元素
        let head_num = 64;
        let head_size = 128;
        let shapes = vec![batch_size, head_num, head_size]; 
        let length = shapes.iter().product(); // 总元素数量
        
        let position_index = 0; // 起始位置，根据实际情况可以修改

        let input_strides1 = get_strides(&shapes);
        let input_strides2 = get_strides(&shapes);
        let output_strides = get_strides(&shapes); 
        
        // 创建模拟的输入和输出数据
        //let input_data: Vec<f32> = (0..length).map(|x| x as f32).collect();
        let input_data1: Vec<f32> = (0..head_size).cycle().take(length).map(|x| x as f32).collect();
        let input_data2:Vec<f32> =vec![1.0;length];
        let results: Vec<f32>=(1..=head_size).cycle().take(length).map(|x| x as f32).collect();
        //println!("{:?}", input_data2);
        let mut output_data: Vec<f32> = vec![0.0; length];
        
            // 使用 chunk_map 函数创建块
        let chunks = chunk_zipmap(shapes,  input_data1.as_ptr(),input_strides1,input_data2.as_ptr(),input_strides2, output_data.as_mut_ptr(),output_strides);
            // 使用这些块和长度初始化 ArgmaxMap
        let thread_num: usize = num_cpus::get();
        let mut argmax_operator = AddZipMap::new(head_size,head_num, thread_num);
        
        argmax_operator.set_chunk(chunks);
        
        for i in 0..thread_num {
            argmax_operator.run(batch_size, position_index,i);
        }
            
            // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[0..length], results[0..length], max_ulps=4); 
        // println!("{:?}", output_data);
        // println!("{:?}", output);
    }
}
