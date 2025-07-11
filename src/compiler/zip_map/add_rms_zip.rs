use std::f16;
// use num_traits::Float;

use crate::kernel::generic::sqrt::Sqrt;
use super::zip_map_trait::ZipMapTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;



#[derive(Clone)]
pub struct AddRMSZipMap<T> {
    chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>,
    length: usize,
    weight: ConstPtr<T>,
    eps: T,
    cpu_num: usize,
}

impl<T> AddRMSZipMap<T> 
where T: Sqrt
{
    pub fn new(length: usize, weight: *const T, eps: T, cpu_num: usize) -> Self {
        Self {
            chunks: vec![],
            length: length,
            weight: ConstPtr { ptr: weight },
            eps: eps,
            cpu_num: cpu_num,
        }
    }

    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }

    pub fn run(&self, batch_size: usize, thread_id: usize) {
        if let Some((begin, end)) = assign(batch_size, self.cpu_num, thread_id) {
            for &(a, b, c) in self.chunks.get(begin..end).unwrap() {
                self.compute(a.ptr, b.ptr, c.ptr);
            }
        }
    }
}
// unsafe impl<T: Float + FromPrimitive> Send for AddRMSZipMap<T> {}
// unsafe impl<T: Float + FromPrimitive> Sync for AddRMSZipMap<T> {}

impl<T> ZipMapTrait<T> for AddRMSZipMap<T>
where T: Sqrt
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        kernel::generic::rms_norm::add_rms_norm(
            input_ptr1,
            input_ptr2,
            output_ptr,
            self.length,
            self.weight.ptr,
            self.eps,
        );
    }
}

impl ZipMapTrait<f16> for AddRMSZipMap<f16> {
    fn compute(&self, input_ptr1: *const f16, input_ptr2: *const f16, output_ptr: *mut f16) {
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::x86_64::f16_512::rms_norm::add_rms_norm(
            input_ptr1,
            input_ptr2,
            output_ptr,
            self.length,
            self.weight.ptr,
            self.eps,
        );

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::rms_norm::add_rms_norm(
            input_ptr1,
            input_ptr2,
            output_ptr,
            self.length,
            self.weight.ptr,
            self.eps,
        );
    }
}

impl ZipMapTrait<f32> for AddRMSZipMap<f32> {
    fn compute(&self, input_ptr1: *const f32, input_ptr2: *const f32, output_ptr: *mut f32) {
    
        kernel::generic::rms_norm::add_rms_norm(
            input_ptr1,
            input_ptr2,
            output_ptr,
            self.length,
            self.weight.ptr,
            self.eps,
        ); 
    }
}
impl ZipMapTrait<f64> for AddRMSZipMap<f64> {
    fn compute(&self, input_ptr1: *const f64, input_ptr2: *const f64, output_ptr: *mut f64) {
        /*
        add_rms_norm_block(
            input_ptr1,
            input_ptr2,
            output_ptr,
            self.length,
            self.weight.ptr,
            self.eps,
        ); */
    }
}
#[cfg(test)]
mod test {
    use super::super::chunk_zipmap::chunk_zipmap;
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_add_rms_zip() {
        let shapes = vec![10, 18];
        let input_strides1 = vec![18, 1];
        let input_strides2 = vec![18, 1];
        let output_strides = vec![18, 1];
        let length = shapes.iter().product(); // 总元素数量
        let batch_size = 10; // 每次批处理 10 个元素
        let position_size = 0; // 起始位置，根据实际情况可以修改

        // 创建模拟的输入和输出数据
        //let input_data: Vec<f32> = (0..length).map(|x| x as f32).collect();
        let input_data1: Vec<f32> = (0..=17).cycle().take(180).map(|x| x as f32).collect();
        let input_data2 = [1.0f32; 180];
        let weight = [1.0f32; 180];
        let eps = 1e-6;
        let mut output_data: Vec<f32> = vec![0.0; length];

        // 使用 chunk_map 函数创建块
        let chunks = chunk_zipmap(
            shapes,
            input_data1.as_ptr(),
            input_strides1,
            input_data2.as_ptr(),
            input_strides2,
            output_data.as_mut_ptr(),
            output_strides,
        );
        // 使用这些块和长度初始化 ArgmaxMap
        let thread_num: usize = num_cpus::get();
        let mut argmax_operator = AddRMSZipMap::new(18, weight.as_ptr(), eps, thread_num);
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

        for i in 0..thread_num {
            argmax_operator.run(batch_size, i);
        }

        // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        println!("{:?}", output_data);
    }
}
