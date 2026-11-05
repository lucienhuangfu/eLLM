use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::kernel;
use crate::num_traits::Sigmoid;
use crate::operators::assign::assign;
use crate::operators::elementwise::traits::ZipMapTrait;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};

// Elementwise zip of an activation and its gate: `out = value * sigmoid(gate)`.
// 逐元素拉链：`out = value * sigmoid(gate)`。
//
// This is the attention output gate of `Qwen3_5Attention.forward`
// (`attn_output = attn_output * torch.sigmoid(gate)`, modeling_qwen3_5.py L808),
// consumed by `FullAttention` when `config.attn_output_gate` is set. Shape and
// chunking contract match `AddZipMap` / `SiluMulZipMap`: `head_num * head_size`
// chunks, `head_size` contiguous elements per chunk (one attention head).
// 对应 Qwen3.5 全注意力输出门；分块契约与 AddZipMap / SiluMulZipMap 一致
// （head_num 个 chunk，每 chunk head_size 个连续元素 = 一个注意力头）。
#[derive(Clone)]
pub struct SigmoidMulZipMap<T> {
    value_ptr: ConstPtr<T>,
    gate_ptr: ConstPtr<T>,
    output_ptr: MutPtr<T>,
    head_num: usize,
    head_size: usize,
}

impl<T> SigmoidMulZipMap<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sigmoid,
{
    /// `head_num * head_size` 个 chunk、每 chunk `head_size` 个连续元素；
    /// 活跃行数由调度器传入的 `total_size` 给出，故此处无需 `decode_only_flag`。
    pub fn new(
        value_ptr: *const T,
        gate_ptr: *const T,
        output_ptr: *mut T,
        head_num: usize,
        head_size: usize,
    ) -> Self {
        Self {
            value_ptr: ConstPtr { ptr: value_ptr },
            gate_ptr: ConstPtr { ptr: gate_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            head_num,
            head_size,
        }
    }

    pub fn run(&self, prefill_size: usize, _total_size: usize, cpu_num: usize, thread_id: usize) {
        self.run_scheduled(prefill_size, 0, _total_size, cpu_num, thread_id);
    }

    pub fn run_scheduled(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        total_size: usize,
        cpu_num: usize,
        thread_id: usize,
    ) {
        let total_len = total_size * self.head_num;
        if let Some((begin, end)) = assign(total_len, cpu_num, thread_id) {
            let value_ptr = self.value_ptr.ptr;
            let gate_ptr = self.gate_ptr.ptr;
            let output_ptr = self.output_ptr.ptr;

            for index in begin..end {
                unsafe {
                    let p = index * self.head_size;
                    self.compute(value_ptr.add(p), gate_ptr.add(p), output_ptr.add(p));
                }
            }
        }
    }
}

impl<T> ZipMapTrait<T> for SigmoidMulZipMap<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Sigmoid,
{
    default fn compute(&self, input_ptr1: *const T, input_ptr2: *const T, output_ptr: *mut T) {
        kernel::scalar::sigmoid::sigmoid_multiply(
            input_ptr1,
            input_ptr2,
            output_ptr,
            self.head_size,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_sigmoid_mul_zip() {
        let batch_size = 2;
        let head_num = 4;
        let head_size = 8;
        let length = batch_size * head_num * head_size;

        let value: Vec<f32> = (0..length).map(|x| x as f32 - 4.0).collect();
        let gate: Vec<f32> = (0..length).map(|x| ((x % 5) as f32) - 2.0).collect();
        let mut output = vec![0.0f32; length];

        let operator = SigmoidMulZipMap::new(
            value.as_ptr(),
            gate.as_ptr(),
            output.as_mut_ptr(),
            head_num,
            head_size,
        );

        let thread_num = 4;
        for i in 0..thread_num {
            operator.run(batch_size, batch_size, thread_num, i);
        }

        for (i, v) in output.iter().enumerate() {
            assert_ulps_eq!(
                *v,
                value[i] * (1.0 / (1.0 + (-gate[i]).exp())),
                max_ulps = 4
            );
        }
    }
}
