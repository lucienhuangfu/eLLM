use std::f16;

use super::map_trait::TopKSoftmaxTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::kernel::generic::from_usize::FromUsize;
use crate::kernel::generic::sqrt::Sqrt;

#[derive(Clone)]
pub struct TopKSoftmax<T> {
    input_indices_ptr: ConstPtr<usize>,
    input_values_ptr: ConstPtr<T>,
    sums_ptr: ConstPtr<T>,
    output_indices_ptr: MutPtr<usize>,
    output_values_ptr: MutPtr<T>,
    batch_size: usize,
    topk_size: usize,
}

impl<T: Sqrt> TopKSoftmax<T> {
    pub fn new(
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        sums_ptr: *const T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        batch_size: usize,
        topk_size: usize,
    ) -> Self {
        Self {
            input_indices_ptr: ConstPtr { ptr: input_indices_ptr },
            input_values_ptr: ConstPtr { ptr: input_values_ptr },
            sums_ptr: ConstPtr { ptr: sums_ptr },
            output_indices_ptr: MutPtr { ptr: output_indices_ptr },
            output_values_ptr: MutPtr { ptr: output_values_ptr },
            batch_size,
            topk_size,
        }
    }

    pub fn run(
        &self,
        position_begin: usize,
        position_interval: usize,
        batch_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
    
        if let Some((begin, end)) = assign(batch_size * position_interval, thread_num, thread_id)
        {
            let (mut row_index, mut col_index) = (begin / batch_size, begin % batch_size);
            let mut input_indices_ptr = self.input_indices_ptr.ptr;
            let mut input_values_ptr = self.input_values_ptr.ptr;
            let mut sums_ptr = self.sums_ptr.ptr;
            let mut output_indices_ptr = self.output_indices_ptr.ptr;
            let mut output_values_ptr = self.output_values_ptr.ptr;

            for _ in begin..end {
                let index = row_index * self.batch_size + col_index;
                unsafe {
                    let input_stride = index * self.topk_size * thread_num;
                    let output_stride = index * self.topk_size;
                    self.compute(input_indices_ptr.add(input_stride), input_values_ptr.add(input_stride), sums_ptr.add(index), output_indices_ptr.add(output_stride), output_values_ptr.add(output_stride), thread_num, self.topk_size);
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

impl<T: Sqrt> TopKSoftmaxTrait<T> for TopKSoftmax<T> {
    default fn compute(&self, input_indices_ptr: *const usize, input_values_ptr: *const T, sums_ptr: *const T, output_indices_ptr: *mut usize, output_values_ptr: *mut T, thread_num: usize, topk_size: usize) {
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

impl TopKSoftmaxTrait<f16> for TopKSoftmax<f16> {
    fn compute(&self, input_indices_ptr: *const usize, input_values_ptr: *const f16, sums_ptr: *const f16, output_indices_ptr: *mut usize, output_values_ptr: *mut f16, thread_num: usize, topk_size: usize) {
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

impl TopKSoftmaxTrait<f32> for TopKSoftmax<f32> {
    fn compute(&self, input_indices_ptr: *const usize, input_values_ptr: *const f32, sums_ptr: *const f32, output_indices_ptr: *mut usize, output_values_ptr: *mut f32, thread_num: usize, topk_size: usize) {
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


#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use num_cpus;
    // use std::ptr;
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
