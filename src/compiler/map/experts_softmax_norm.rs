use std::f16;

use super::map_trait::SoftmaxTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::kernel::generic::from_usize::FromUsize;
use crate::kernel::generic::sqrt::Sqrt;

#[derive(Clone)]
pub struct ExpertsSoftmaxNorm<T> {
    ptr1: ConstPtr<T>,
    sum_ptr: ConstPtr<T>,
    max_ptr: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    batch_size: usize,
    topk_size: usize,
}

impl<T: Sqrt> ExpertsSoftmaxNorm<T> {
    pub fn new(
        ptr1: *const T,
        sum_ptr: *const T,
        output_ptr: *mut T,
        max_ptr: *const T,
        batch_size: usize,
        topk_size: usize,
    ) -> Self {
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            sum_ptr: ConstPtr { ptr: sum_ptr },
            max_ptr: ConstPtr { ptr: max_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            batch_size,
            topk_size,
        }
    }

    pub fn run(
        &self,
        position_begin: usize,
        position_interval: usize,
        batch_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        /*
        if let Some((begin, end)) = assign(batch_size * position_interval, cpu_num, thread_id)
        {
            let (mut row_index, mut col_index) = (begin / batch_size, begin % batch_size);
            let mut ptr1 = self.ptr1.ptr;
            let mut output_ptr = self.output_ptr.ptr;

            for _ in begin..end {
                let index = row_index * self.max_batch_size + col_index;
                unsafe {
                    // let (a, b) = self.chunks.get_unchecked(index);
                    let p = index * self.hidden_size;
                    self.compute(ptr1.add(p), output_ptr.add(p), self.hidden_size);
                }
                col_index += 1;
                if col_index == batch_size {
                    col_index = 0;
                    row_index += 1;
                }
            }
        } */
    }
}

impl<T: Sqrt> SoftmaxTrait<T> for ExpertsSoftmaxNorm<T> {
    default fn compute(&self, input_ptr: *const T, sum_ptr: *const T, max_ptr: *const T, output_ptr: *mut T, length: usize) {
        /* 
        kernel::generic::softmax::softmax(
            input_ptr,
            sum_ptr.ptr,
            max_ptr.ptr,
            output_ptr,
            length,
        );*/
    }
}

impl SoftmaxTrait<f16> for ExpertsSoftmaxNorm<f16> {
    fn compute(&self, input_ptr: *const f16, sum_ptr: *const f16, max_ptr: *const f16, output_ptr: *mut f16, length: usize) {
        /* 
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::softmax::softmax(
            input_ptr,
            sum_ptr.ptr,
            max_ptr.ptr,
            output_ptr,
            length,
        );

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::softmax::softmax(
            input_ptr,
            sum_ptr.ptr,
            max_ptr.ptr,
            output_ptr,
            length,
        );*/
    }
}

impl SoftmaxTrait<f32> for ExpertsSoftmaxNorm<f32> {
    fn compute(&self, input_ptr: *const f32, sum_ptr: *const f32, max_ptr: *const f32, output_ptr: *mut f32, length: usize) {
        /*
        kernel::generic::softmax::softmax(
            input_ptr,
            sum_ptr.ptr,
            max_ptr.ptr,
            output_ptr,
            length,
        ); */
    }
}

impl SoftmaxTrait<f64> for ExpertsSoftmaxNorm<f64> {
    fn compute(&self, input_ptr: *const f64, sum_ptr: *const f64, max_ptr: *const f64, output_ptr: *mut f64, length: usize) {
        kernel::generic::softmax::softmax(
            input_ptr,
            sum_ptr.ptr,
            max_ptr.ptr,
            output_ptr,
            length,
        );
    }
}

#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use num_cpus;
    // use std::ptr;

    // use super::super::chunk_map::chunk_map;
    // use crate::memory::allocator::allocate_init;
    use super::*;

    /*
    #[test]
    fn test_rms_map() {
        let seq_threshold = 64; // 序列长度
        let batch_size = 10; // 每个批次处理 10 个元素
        let hidden_size = 18;

        let shapes = vec![seq_threshold, batch_size, hidden_size];
        let strides = vec![batch_size * hidden_size, hidden_size, 1]; // 对应的步长
        let length = shapes.iter().product(); // 总元素数量
                                              // let batch_size = 10; // 每次批处理 10 个元素
        let position_index = 0; // 起始位置，根据实际情况可以修改
        let position_interval = 4; // 间隔位置，根据实际情况可以修改

        let cpu_num = num_cpus::get();

        // 创建模拟的输入和输出数据
        let input_data: Vec<f32> = (1..=18).cycle().take(180).map(|x| x as f32).collect();
        let weight = vec![1.0f32; hidden_size];
        let eps = 1e-6;
        let mut output_data: Vec<f32> = vec![0.0; length];

        /*
        // 使用 chunk_map 函数创建块
        let chunks = chunk_map(
            shapes,
            strides,
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
        );
         */

        // 使用这些块和长度初始化 ArgmaxMap
        let mut operator = RMSMap::new(
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
            batch_size,
            hidden_size,
            weight.as_ptr(),
            eps,
            // cpu_num,
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
            operator.run(position_index, position_interval, batch_size,  thread_num, i);
        }
        // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[18..36], result, max_ulps = 4);
        // println!("{:?}", output_data);
    }*/
}
