use std::f16;

use super::map_trait::MapTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
// use crate::kernel::generic::from_usize::FromUsize;
use crate::kernel::generic::sqrt::Sqrt;

#[derive(Clone)]
pub struct RMSMap<T> {
    ptr1: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    hidden_size: usize,
    eps: T,
    decode_only_flag: bool,
}

impl<T: Sqrt> RMSMap<T> {
    pub fn new(ptr1: *const T, output_ptr: *mut T, hidden_size: usize, eps: T, decode_only_flag: bool) -> Self {
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            output_ptr: MutPtr { ptr: output_ptr },
            hidden_size,
            eps: eps,
            decode_only_flag,
        }
    }

    pub fn run(&self, token_size: usize, decode_size: usize, thread_num: usize, thread_id: usize) {
        let task_size = if self.decode_only_flag == true {
            decode_size
        } else {
            token_size
        };

        if let Some((begin, end)) = assign(task_size, thread_num, thread_id) {
            let mut ptr1 = self.ptr1.ptr;
            let mut output_ptr = self.output_ptr.ptr;

            for index in begin..end {
                unsafe {
                    let p = index * self.hidden_size;
                    self.compute(ptr1.add(p), output_ptr.add(p), self.hidden_size);
                }
            }
        }
    }
}

impl<T: Sqrt> MapTrait<T> for RMSMap<T> {
    default fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize) {
        kernel::generic::rms_norm::rms_norm(
            input_ptr, output_ptr, length, // self.weight.ptr,
            self.eps,
        );
    }
}

impl MapTrait<f16> for RMSMap<f16> {
    fn compute(&self, input_ptr: *const f16, output_ptr: *mut f16, length: usize) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::rms_norm::rms_norm(
            input_ptr, output_ptr, length, // self.weight.ptr,
            self.eps,
        );

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::rms_norm::rms_norm(
            input_ptr, output_ptr, length, // self.weight.ptr,
            self.eps,
        );
    }
}

impl MapTrait<f32> for RMSMap<f32> {
    fn compute(&self, input_ptr: *const f32, output_ptr: *mut f32, length: usize) {
        kernel::generic::rms_norm::rms_norm(
            input_ptr, output_ptr, length, // self.weight.ptr,
            self.eps,
        );
    }
}

impl MapTrait<f64> for RMSMap<f64> {
    fn compute(&self, input_ptr: *const f64, output_ptr: *mut f64, length: usize) {
        kernel::generic::rms_norm::rms_norm(
            input_ptr, output_ptr, length, // self.weight.ptr,
            self.eps,
        );
    }
}

#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use num_cpus;

    use super::*;

    #[test]
    fn test_rms_map() {
        let batch_size = 10; // 每个批次处理 10 个元素
        let hidden_size = 18;

        let shapes = vec![batch_size, hidden_size];
        // let strides = vec![batch_size * hidden_size, hidden_size, 1]; // 对应的步长
        let length = shapes.iter().product(); // 总元素数量
                                              // let batch_size = 10; // 每次批处理 10 个元素

        let cpu_num = num_cpus::get();

        // 创建模拟的输入和输出数据
        let input_data: Vec<f32> = (1..=18).cycle().take(180).map(|x| x as f32).collect();
        let weight = vec![1.0f32; hidden_size];
        let eps = 1e-6f32;
        let mut output_data: Vec<f32> = vec![0.0; length];


        // 使用这些块和长度初始化 ArgmaxMap
        let mut operator = RMSMap::new(
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
            hidden_size,
            // weight.as_ptr(),
            eps,
            false
        );
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
        // operator.set_chunk(chunks);
        let thread_num: usize = cpu_num;

        for i in 0..thread_num {
            operator.run(batch_size, 0, thread_num, i);
        }
        // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        // println!("{:?}", output_data);
    }

    /*
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
    }*/
}
