use std::f16;
use std::ops::{AddAssign, Sub};
use super::map_trait::SoftmaxTrait;
use crate::compiler::assign::assign;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel::generic;
use crate::memory::allocator::allocate_init;
use crate::kernel::generic::{exp::Exp, sqrt::Sqrt};
use crate::kernel::x86_64;

#[derive(Clone)]
pub struct ExpertsSoftmaxNorm<T> {
    // [sequence_chunk_size, batch_size, num_experts]
    ptr1: ConstPtr<T>,
    topk_values_ptr: *mut T,
    topk_indices_ptr: *mut usize,
    // Expert routing information
    experts_indicator: MutPtr<bool>,
    indice_ptr: MutPtr<bool>,
    weight_ptr: MutPtr<T>,

    num_tokens: usize,
    batch_size: usize,
    num_experts: usize,
    num_topk: usize,
}

impl<T: Sqrt + Default> ExpertsSoftmaxNorm<T> {
    pub fn new(
        ptr1: *const T,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        weight_ptr: *mut T,
        sequence_chunk_size: usize,
        batch_size: usize,
        num_experts: usize,
        num_topk: usize,
    ) -> Self {
        let length = sequence_chunk_size * batch_size * num_topk;
        Self {
            ptr1: ConstPtr { ptr: ptr1 },

            topk_values_ptr: unsafe {
                allocate_init::<T>(length, T::default())
            },
            topk_indices_ptr: unsafe {
                allocate_init::<usize>(length, 0)
            },
            experts_indicator: MutPtr {
                ptr: experts_indicator,
            },
            indice_ptr: MutPtr { ptr: indice_ptr },
            weight_ptr: MutPtr { ptr: weight_ptr },
            num_tokens: sequence_chunk_size * batch_size,
            batch_size,
            num_experts,
            num_topk,
        }
    }
}

impl<T: Sqrt + Exp + Default + AddAssign + Sub<Output = T> + Copy> ExpertsSoftmaxNorm<T> {
    pub fn run(
        &self,
        position_begin: usize,
        position_interval: usize,
        batch_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        if let Some((begin, end)) = assign(batch_size * position_interval, thread_num, thread_id) {
            let (mut row_index, mut col_index) = (begin / batch_size, begin % batch_size);

            let ptr1 = self.ptr1.ptr;

            for _ in begin..end {
                let index = row_index * self.batch_size + col_index;
                unsafe {
                    let p1 = index * self.num_experts;
                    let p2 = index * self.num_topk;
                    let token_index = col_index + row_index * batch_size;
                    self.compute(
                        ptr1.add(p1),
                        self.topk_values_ptr.add(p2),
                        self.topk_indices_ptr.add(p2),
                        self.experts_indicator.ptr,
                        self.indice_ptr.ptr,
                        self.weight_ptr.ptr,
                        token_index,
                        self.num_experts,
                        self.num_topk,
                    );
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

impl<T: Sqrt + Exp + Default + AddAssign + Sub<Output = T> + Copy> SoftmaxTrait<T>
    for ExpertsSoftmaxNorm<T>
{
    default fn compute(
        &self,
        input_ptr: *const T,
        topk_values_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        weight_ptr: *mut T,
        token_index: usize,
        input_length: usize,
        output_length: usize,
    ) {
        generic::experts_topk_softmax_norm::experts_topk_softmax_norm(
            input_ptr,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            token_index,
            self.num_tokens,
            input_length,
            output_length,
        );
    }
}

impl SoftmaxTrait<f16> for ExpertsSoftmaxNorm<f16> {
    fn compute(
        &self,
        input_ptr: *const f16,
        topk_values_ptr: *mut f16,
        topk_indices_ptr: *mut usize,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        weight_ptr: *mut f16,
        token_index: usize,
        input_length: usize,
        output_length: usize,
    ) {
        /* 
        x86_64::f16_512::experts_topk_softmax_norm::experts_topk_softmax_norm(
            input_ptr,
            topk_values_ptr,
            topk_indices_ptr,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            token_index,
            self.num_tokens,
            input_length,
            output_length,
        );*/
    }
}

impl SoftmaxTrait<f32> for ExpertsSoftmaxNorm<f32> {
    fn compute(
        &self,
        input_ptr: *const f32,
        topk_values_ptr: *mut f32,
        topk_indices_ptr: *mut usize,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        weight_ptr: *mut f32,
        token_index: usize,
        input_length: usize,
        output_length: usize,
    ) {
        x86_64::f32_256::experts_topk_softmax_norm::experts_topk_softmax_norm(
            input_ptr,
            topk_values_ptr,
            topk_indices_ptr,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            token_index,
            self.num_tokens,
            input_length,
            output_length,
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

    #[test]
    fn test_experts_softmax_norm_f32() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping test.");
            return;
        }

        let sequence_chunk_size = 1;
        let batch_size = 2;
        let num_experts = 16;
        let num_topk = 4;
        let num_tokens = sequence_chunk_size * batch_size;

        let input_data1: Vec<f32> = vec![
            0.5, -1.0, 2.5, 3.0, 7.5, 6.5, -2.0, 10.0, 4.0, 8.0, 1.0, 9.5, -3.5, 5.5, 11.0, -0.25,
        ];
        let input_data2: Vec<f32> = vec![
            -0.5, 0.25, 3.75, -2.0, 6.0, 1.75, -4.25, 2.5, 0.0, 5.25, -1.25, 4.0, 3.0, -3.5, 7.5,
            2.25,
        ];
        let mut input_data = Vec::new();
        input_data.extend_from_slice(&input_data1);
        input_data.extend_from_slice(&input_data2);

        let mut experts_indicator = vec![false; num_experts];
        let mut indice_ptr = vec![false; num_experts * num_tokens];
        let mut weight_ptr = vec![0.0f32; num_experts * num_tokens];

        let operator = ExpertsSoftmaxNorm::<f32>::new(
            input_data.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indice_ptr.as_mut_ptr(),
            weight_ptr.as_mut_ptr(),
            sequence_chunk_size,
            batch_size,
            num_experts,
            num_topk,
        );

        let thread_num = 1;
        let thread_id = 0;
        operator.run(0, sequence_chunk_size, batch_size, thread_num, thread_id);

        // Verification for token 0
        let mut expected1: Vec<(usize, f32)> = input_data1.iter().copied().enumerate().collect();
        expected1.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_val1 = input_data1
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denom1: f32 = input_data1.iter().map(|v| (v - max_val1).exp()).sum();

        for i in 0..num_topk {
            let (idx, val) = expected1[i];
            let prob = (val - max_val1).exp() / denom1;
            assert!(experts_indicator[idx]);
            let offset = idx * num_tokens + 0;
            assert!(indice_ptr[offset]);
            assert_ulps_eq!(weight_ptr[offset], prob, max_ulps = 4);
        }

        // Verification for token 1
        let mut expected2: Vec<(usize, f32)> = input_data2.iter().copied().enumerate().collect();
        expected2.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_val2 = input_data2
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let denom2: f32 = input_data2.iter().map(|v| (v - max_val2).exp()).sum();

        for i in 0..num_topk {
            let (idx, val) = expected2[i];
            let prob = (val - max_val2).exp() / denom2;
            assert!(experts_indicator[idx]);
            let offset = idx * num_tokens + 1;
            assert!(indice_ptr[offset]);
            assert_ulps_eq!(weight_ptr[offset], prob, max_ulps = 4);
        }
    }

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
