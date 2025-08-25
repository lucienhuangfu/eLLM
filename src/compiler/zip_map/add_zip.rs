
use std::f16;
use std::ops::{Add,Sub, Mul, Div, Neg};

use axum::http::HeaderName;

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use super::zip_map_trait::ZipMapTrait;
use crate::compiler::assign::assign;
use crate::kernel;


#[derive(Clone)]
pub struct AddZipMap <T> {
    // chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    ptr1: ConstPtr<T>,
    ptr2: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    max_batch_size: usize,
    head_num: usize,
    head_size: usize,
    cpu_num: usize
}

impl <T> AddZipMap <T> 
where T: Copy + Default + Add<Output = T> + Sub<Output = T>+ Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,
{
    pub fn new(
        ptr1: *const T,
        ptr2: *const T,
        output_ptr: *mut T,
        max_batch_size: usize,
        head_num: usize,
        head_size: usize,
        cpu_num: usize
    ) -> Self {
        Self {
            // chunks: vec![],
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: ptr2 },
            output_ptr: MutPtr { ptr: output_ptr },
            max_batch_size,
            head_num,
            head_size,
            cpu_num
        }
    }

    /*
    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }
     */
    pub fn run(&self,   
            position_begin: usize,
            position_interval: usize, 
            batch_size: usize, 
            thread_id: usize) {

        //     [sequence, batch_size, head_num]
        // stride [batch_size * head_num, head_num,1]
        let stride = batch_size * self.head_num;
        let max_stride = self.max_batch_size * self.head_num;
        if let Some((begin, end)) = assign(position_interval * stride, self.cpu_num, thread_id) {
      
            // 从begin得到对应的坐标
            let (mut high_index, mut _index) = (begin / stride, begin % stride);
            let (mut row_index, mut col_index) = (_index / self.head_num, _index % self.head_num);

            println!("thread_id: {}, begin: {}, end: {}, high_index: {}, row_index: {}, col_index: {}", thread_id, begin, end, high_index, row_index, col_index);

            let ptr1 = self.ptr1.ptr;
            let ptr2 = self.ptr2.ptr;
            let output_ptr = self.output_ptr.ptr;

            // 遍历每个chunk
            for i in begin..end {

                println!(" high_index: {}, row_index: {}, col_index: {}",  high_index, row_index, col_index);
                let index = high_index* max_stride + row_index * self.head_num + col_index;
                unsafe {
                    // let (a, b, c) = self.chunks.get_unchecked(index);
                    let p = index * self.head_size;
                    self.compute(ptr1.add(p), ptr2.add(p), output_ptr.add(p));
                }
                col_index += 1;
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

// unsafe impl <T:Float>Send for AddZipMap<T> {}
// unsafe impl <T:Float>Sync for AddZipMap<T> {}
impl <T> ZipMapTrait <T> for AddZipMap  <T>
where T: Copy + Default + Add<Output = T>+ Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + Sigmoid<T>,
{
    
    default fn compute(&self, input_ptr1: *const T, input_ptr2:*const T, output_ptr: *mut T, ) {
        // print!("generic \n");
        kernel::generic::add::add(input_ptr1 , input_ptr2 , output_ptr , self.head_size);
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
        kernel::generic::add::add(input_ptr1 , input_ptr2, output_ptr , self.head_size);
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
        kernel::generic::add::add(input_ptr1, input_ptr2, output_ptr , self.head_size);
    }
}




#[cfg(test)]
mod test {
    
    use super::*;
    use super::super::chunk_zipmap::chunk_zipmap;
    use crate::ptensor::tensor_utils::get_strides;
    use approx::assert_ulps_eq;
    // use nom::sequence;
    // use rand::seq;

    #[test]
    fn test_add_zip() {
        let sequence_threshold = 4;
        let batch_size = 2; // 每次批处理 10 个元素
        let head_num = 8;
        let head_size = 4;
        let shapes = vec![sequence_threshold, batch_size, head_num, head_size]; 
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
        // println!("{:?}", input_data2);
        let mut output_data: Vec<f32> = vec![0.0; length];
        
        // 使用 chunk_map 函数创建块
        // let chunks = chunk_zipmap(shapes,  input_data1.as_ptr(),input_strides1,input_data2.as_ptr(),input_strides2, output_data.as_mut_ptr(),output_strides);
        // 使用这些块和长度初始化 ArgmaxMap
        let thread_num: usize = 4; 
        // num_cpus::get();


        let mut operator = AddZipMap::new(input_data1.as_ptr(), input_data2.as_ptr(), output_data.as_mut_ptr(), batch_size, head_num, head_size, thread_num);

        // operator.set_chunk(chunks);
        
        for i in 0..thread_num {
            operator.run(position_index, sequence_threshold, batch_size, i);
        }
            
            // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[0..sequence_threshold*batch_size*head_num*head_size], results[0..sequence_threshold*batch_size*head_num*head_size], max_ulps=4); 
        // println!("{:?}", output_data[0..batch_size*head_num*head_size]);
        // println!("{:?}", output);
    }
}
