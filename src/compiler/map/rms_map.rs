use std::f16;

use crate::kernel;
use crate::kernel::generic::from_usize::FromUsize;
use crate::kernel::generic::sqrt::Sqrt;
use super::map_trait::MapTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};

#[derive(Clone)]
pub struct RMSMap<T> {
    chunks: Vec<(ConstPtr<T>, MutPtr<T>)>,
    length: usize,
    max_batch_size: usize,
    weight: ConstPtr<T>,
    eps: T,
    cpu_num: usize,
}

impl<T: Sqrt> RMSMap<T> {
    pub fn new(max_batch_size: usize, length: usize, weight: *const T, eps: T, cpu_num: usize) -> Self {
        Self {
            chunks: vec![],
            max_batch_size,
            length,
            weight: ConstPtr { ptr: weight },
            eps: eps,
            cpu_num: cpu_num,
        }
    }

    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }

    pub fn run(&self, batch_size: usize, position_begin: usize, position_interval: usize, thread_id: usize) {
        if let Some((begin, end)) = assign(batch_size * position_interval, self.cpu_num, thread_id) {

            let (mut row_index, mut col_index) = (begin / batch_size, begin % batch_size);
            for i in begin..end {
                let index = row_index * self.max_batch_size + col_index; 
                unsafe {
                     let (a, b) = self.chunks.get_unchecked(index);
                    self.compute(a.ptr, b.ptr, self.length);
                }
                col_index += 1;
                if col_index == batch_size {
                    col_index = 0;
                    row_index += 1;
                }
            }
        }
    }
}

impl<T: Sqrt> MapTrait<T> for RMSMap<T> {
    default fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize) {
        kernel::generic::rms_norm::rms_norm(
            input_ptr,
            output_ptr,
            length,
            self.weight.ptr,
            self.eps,
        );
    }
}

impl MapTrait<f16> for RMSMap<f16> {
    fn compute(&self, input_ptr: *const f16, output_ptr: *mut f16, length: usize) {
        
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::rms_norm::rms_norm(
            input_ptr,
            output_ptr,
            length,
            self.weight.ptr,
            self.eps,
        );
        
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::rms_norm::rms_norm(
            input_ptr,
            output_ptr,
            length,
            self.weight.ptr,
            self.eps,
        );
    }
}

impl MapTrait<f32> for RMSMap<f32> {
    fn compute(&self, input_ptr: *const f32, output_ptr: *mut f32, length: usize) {
        kernel::generic::rms_norm::rms_norm(
            input_ptr,
            output_ptr,
            length,
            self.weight.ptr,
            self.eps,
        );
    }
}

impl MapTrait<f64> for RMSMap<f64> {
    fn compute(&self, input_ptr: *const f64, output_ptr: *mut f64, length: usize) {
        kernel::generic::rms_norm::rms_norm(
            input_ptr,
            output_ptr,
            length,
            self.weight.ptr,
            self.eps,
        );
    }
}

#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use num_cpus;
    use std::ptr;

    use super::super::chunk_map::chunk_map;
    use crate::memory::allocator::allocate_init;
    use super::*;

    #[test]
    fn test_rms_map() {
        let shapes = vec![10, 18];
        let strides = vec![18, 1]; // 对应的步长
        let length = shapes.iter().product(); // 总元素数量
        let batch_size = 10; // 每次批处理 10 个元素
        let position_index = 0; // 起始位置，根据实际情况可以修改
        let cpu_num = num_cpus::get();

        // 创建模拟的输入和输出数据
        let input_data: Vec<f32> = (1..=18).cycle().take(180).map(|x| x as f32).collect();
        let weight = [1.0f32; 180];
        let eps = 1e-6;
        let mut output_data: Vec<f32> = vec![0.0; length];

        // 使用 chunk_map 函数创建块
        let chunks = chunk_map(
            shapes,
            strides,
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
        );
        // 使用这些块和长度初始化 ArgmaxMap
        let mut argmax_operator = RMSMap::new(18, weight.as_ptr(), eps, cpu_num);
        let result = [
            0.09238425642251968,
            0.18476851284503937,
            0.27715277671813965,
            0.36953702569007874,
            0.4619212746620178,
            0.5543055534362793,
            0.646689772605896,
            0.7390740513801575,
            0.831458330154419,
            0.9238425493240356,
            1.0162267684936523,
            1.1086111068725586,
            1.2009953260421753,
            1.293379545211792,
            1.3857638835906982,
            1.478148102760315,
            1.5705323219299316,
            1.662916660308838,
        ];
        argmax_operator.set_chunk(chunks);
        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            argmax_operator.run(batch_size, 0, i);
        }
        // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        // println!("{:?}", output_data);
    }

    #[test]
    fn test_rms_map_f16() {
        let batch_size = 10; // 每次批处理 10 个元素
        let hidden_size = 128;

        let shapes = vec![batch_size, hidden_size];
        let strides = vec![hidden_size, 1]; // 对应的步长
        let length = shapes.iter().product(); // 总元素数量
     
        let position_index = 5; 
        let cpu_num = num_cpus::get();
        let eps = 1e-6;
        // 创建模拟的输入和输出数据
        /* 
        let input_data: Vec<f16> = (1..=hidden_size).cycle().take(length).map(|x| x as f16).collect();
        let weight = vec![1.0; hidden_size];
        let mut output_data: Vec<f16> = vec![0.0; length];
        */

        let input_data = allocate_init::<f16>(length, 0.0);
        for i in 0..batch_size {

            for j in 0..hidden_size {
                unsafe {
                    ptr::write(input_data.wrapping_add(i*batch_size + j), j as f16);
                }
            }
        }

        let weight = allocate_init::<f16>(hidden_size, 1.0);
        let output_data = allocate_init::<f16>(length, 1.0);


        // 使用 chunk_map 函数创建块
        let chunks = chunk_map(
            shapes,
            strides,
            input_data,
            output_data,
        );
        // 使用这些块和长度初始化 ArgmaxMap
        let mut argmax_operator = RMSMap::new(hidden_size, weight, eps, cpu_num);

        argmax_operator.set_chunk(chunks);
        let thread_num: usize = cpu_num;
        for i in 0..thread_num {
            argmax_operator.run(batch_size, position_index, i);
        }

        /*
        let result = [
            0.09238425642251968,
            0.18476851284503937,
            0.27715277671813965,
            0.36953702569007874,
            0.4619212746620178,
            0.5543055534362793,
            0.646689772605896,
            0.7390740513801575,
            0.831458330154419,
            0.9238425493240356,
            1.0162267684936523,
            1.1086111068725586,
            1.2009953260421753,
            1.293379545211792,
            1.3857638835906982,
            1.478148102760315,
            1.5705323219299316,
            1.662916660308838,
        ];


        // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        // println!("{:?}", output_data);
         */
    }




}
