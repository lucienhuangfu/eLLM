use super::map_trait::MapTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel::generic::softmax::scale_softmax as softmax_block;
use std::f16;
use num_traits::Float;
use std::ops::AddAssign;

#[derive(Clone)]
pub struct SoftmaxMap<T: Float> {
    chunks: Vec<(ConstPtr<T>, MutPtr<T>)>,
    head_num: usize,
    inverse_sqrt_head: T,
    cpu_num: usize,
}

impl<T: Float + AddAssign> SoftmaxMap<T> {
    pub fn new(
        // chunks: Vec<(ConstPtr<T>, MutPtr<T>)>,
        head_num: usize,
        cpu_num: usize,
    ) -> Self {
        Self {
            chunks: vec![],
            head_num: head_num,
            inverse_sqrt_head: T::from((head_num as f32).sqrt().recip()).unwrap(),
            cpu_num: cpu_num,
        }
    }

    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, MutPtr<T>)>) {
        self.chunks = chunks;
    }

    pub fn run(&self, batch_size: usize, position_index: usize, thread_id: usize) {
        //println!("{:?}",self.inverse_sqrt_head.to_f32().unwrap());
        let length = batch_size * self.head_num;
        if let Some((begin, end)) = assign(length, self.cpu_num, thread_id) {
            // let begin = 0;
            // let end = batch_size * self.head_num;
            for &(a, b) in self.chunks.get(begin..end).unwrap() {
                self.compute(a.ptr, b.ptr, position_index);
            }
        }
    }
}

// unsafe impl <T:Float>Send for SoftmaxMap<T> {}
// unsafe impl <T:Float>Sync for SoftmaxMap<T> {}

impl<T: Float + AddAssign> MapTrait<T> for SoftmaxMap<T> {
    default fn compute(&self, input_ptr: *const T, output_ptr: *mut T, length: usize) {
        //print!("generic runner\n");
        softmax_block(input_ptr, output_ptr, length, self.inverse_sqrt_head);
    }
}

impl MapTrait<f16> for SoftmaxMap<f16> {
    fn compute(&self, input_ptr: *const f16, output_ptr: *mut f16, length: usize) {
        //print!("f16 runner\n");
        /*
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            SIMD_f16_512_softmax_block(
                input_ptr , output_ptr, length,self.inverse_sqrt_head,
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        */
        softmax_block(input_ptr, output_ptr, length, self.inverse_sqrt_head);
    }
}

impl MapTrait<f32> for SoftmaxMap<f32> {
    fn compute(&self, input_ptr: *const f32, output_ptr: *mut f32, length: usize) {
        //print!("f32 runner\n");
        /*#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_softmax_block(
                input_ptr, output_ptr , length,self.inverse_sqrt_head
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        softmax_block(input_ptr, output_ptr, length, self.inverse_sqrt_head);
    }
}
impl MapTrait<f64> for SoftmaxMap<f64> {
    fn compute(&self, input_ptr: *const f64, output_ptr: *mut f64, length: usize) {
        //print!("f64 runner\n");
        softmax_block(input_ptr, output_ptr, length, self.inverse_sqrt_head);
    }
}
#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use num_cpus;

    use super::super::chunk_map::chunk_map;
    use super::*;
    use crate::ptensor::tensor_utils::get_strides;

    #[test]
    fn test_softmax_map() {
        let batch_size = 10;
        let head_num = 10;
        let sequence_size = 18;
        let shapes = vec![batch_size, head_num, sequence_size];
        let strides = get_strides(&shapes);
        let length = shapes.iter().product(); // 总元素数量
        let position_index = 18; // 起始位置，根据实际情况可以修改
        let cpu_num = num_cpus::get();

        // 创建模拟的输入和输出数据
        //let input_data: Vec<f32> = (0..length).map(|x| x as f32).collect();
        let input_data: Vec<f32> = (1..=18).cycle().take(1800).map(|x| x as f32).collect();

        let mut output_data: Vec<f32> = vec![0.0; length];

        // 使用 chunk_map 函数创建块
        let chunks = chunk_map(
            shapes,
            strides,
            input_data.as_ptr(),
            output_data.as_mut_ptr(),
        );
        // 使用这些块和长度初始化 ArgmaxMap
        let mut argmax_operator = SoftmaxMap::new(head_num, cpu_num);
        let result = [
            0.0012586231,
            0.0017267587,
            0.0023690138,
            0.0032501512,
            0.0044590216,
            0.006117522,
            0.00839289,
            0.011514563,
            0.015797319,
            0.02167302,
            0.02973414,
            0.040793534,
            0.0559664,
            0.0767827,
            0.105341434,
            0.14452243,
            0.19827652,
            0.27202395,
        ];
        argmax_operator.set_chunk(chunks);
        let thread_num: usize = cpu_num;
        // 运行 ArgmaxMap 操作
        for i in 0..thread_num {
            argmax_operator.run(batch_size, position_index, i);
        }
        // 如需打印输出数据，请取消以下注释
        assert_ulps_eq!(output_data[0..18], result, max_ulps = 4);
        //println!("{:?}", output_data);

        // println!("{:?}", output);
    }
}
