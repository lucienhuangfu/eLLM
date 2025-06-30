use std::f16;

use super::reduce_trait::ReduceTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;

#[derive(Clone)]
pub struct ArgmaxReduce<T> {
    chunks: Vec<(ConstPtr<T>, MutPtr<usize>)>,
    length: usize,
    cpu_num: usize,
    max_batch_size: usize,
    // sequences: *mut usize
}

impl<T> ArgmaxReduce<T> 
where T: Copy + PartialOrd {
    pub fn new(
        // chunks: Vec<(ConstPtr<T>, MutPtr<T>)>,
        length: usize,
        max_batch_size: usize,
        cpu_num: usize,
        // sequences: *mut T
    ) -> Self {
        Self {
            chunks: vec![],
            length: length,
            max_batch_size: max_batch_size,
            cpu_num: cpu_num,
            // sequences: sequences
        }
    }

    pub fn set_chunk(&mut self, chunks: Vec<(ConstPtr<T>, MutPtr<usize>)>) {
        self.chunks = chunks;
    }

    pub fn run(&self, batch_size: usize, position_index: usize, thread_id: usize) {
        if let Some((begin, end)) = assign(batch_size, self.cpu_num, thread_id) {
            // let mut current = self.sequences.wrapping_add((position_index + 1) * self.max_batch_size);
            let index = position_index * self.max_batch_size;
            for (a, b) in self.chunks.get((index + begin)..(index + end)).unwrap() {
                self.compute(a.ptr, b.ptr, self.length);
            }
        }
    }
}

// unsafe impl <T:Float>Send for ArgmaxReduce<T> {}
// unsafe impl <T:Float>Sync for ArgmaxReduce<T> {}

impl<T> ReduceTrait<T> for ArgmaxReduce<T> 
where T: Copy + PartialOrd,
{
    default fn compute(&self, input_ptr: *const T, output_ptr: *mut usize, length: usize) {
        // print!("generic runner\n");
        kernel::generic::argmax::argmax(input_ptr, output_ptr, length);
    }
}

impl ReduceTrait<f16> for ArgmaxReduce<f16> {
    fn compute(&self, input_ptr: *const f16, output_ptr: *mut usize, length: usize) {
        //print!("f16 runner\n");
        
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        unsafe {
            kernel::x86_64::f16_512::argmax::argmax(
                input_ptr, output_ptr, length,
            );
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::generic::argmax::argmax(input_ptr as *const f32, output_ptr, length);
    }
}

impl ReduceTrait<f32> for ArgmaxReduce<f32> {
    fn compute(&self, input_ptr: *const f32, output_ptr: *mut usize, length: usize) {
        // print!("f32 runner\n");
        /*#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            SIMD_f32_256_argmax_block(
                input_ptr, output_ptr, length,
            );
        };
        // #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]*/
        // argmax_block(input_ptr, output_ptr, length);
    }
}
impl ReduceTrait<f64> for ArgmaxReduce<f64> {
    fn compute(&self, input_ptr: *const f64, output_ptr: *mut usize, length: usize) {
        // print!("f64 runner\n");
        // argmax_block(input_ptr as *const f32, output_ptr, length);
    }
}

#[cfg(test)]
mod test {
    use approx::assert_ulps_eq;
    use num_cpus;

    use super::super::chunk_reduce::chunk_reduce;
    use super::*;

    #[test]
    fn test_argmax_map() {
        let batch_size = 10; // 每次批处理 10 个元素
        let vocab_size = 64;

        let shapes = vec![batch_size, vocab_size]; // 例子中的形状，10 行 5 列
        let strides = vec![vocab_size, 1]; // 对应的步长
        let length: usize = shapes.iter().product(); // 总元素数量

        // 创建模拟的输入和输出数据
        let input_data: Vec<f32> = (1..=vocab_size)
            .cycle()
            .take(vocab_size * batch_size)
            .map(|x| x as f32)
            .collect();
        // let sequences = vec![0.0f32; sequence_length * batch_size];

        let mut output_data: Vec<usize> = vec![0usize; batch_size];
        let output_strides = vec![1usize];
        // 使用 chunk_reduce 函数创建块
        let chunks = chunk_reduce(
            shapes,
            input_data.as_ptr(),
            strides,
            output_data.as_mut_ptr(),
            output_strides,
        );
        // 使用这些块和长度初始化 ArgmaxReduce

        let thread_num: usize = num_cpus::get();
        let mut argmax_operator = ArgmaxReduce::new(vocab_size, batch_size, thread_num);
        argmax_operator.set_chunk(chunks);

        for i in 0..thread_num {
            argmax_operator.run(batch_size, 0, i);
        }

        // 如需打印输出数据，请取消以下注释
        println!("{:?}", output_data);
    }
}
