use std::marker::PhantomData;

use crate::num_traits::{Sigmoid, Sqrt};
use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::ZipMapTrait;

// Fused gated RMSNorm for GatedDeltaNet-style linear attention layers.
// One kernel replaces the separate normalization and gating steps on the
// recurrent output and the z gate branch:
//   variance = mean(x^2) over the last dim
//   out[i] = weight[i] * x[i] * rsqrt(variance + eps) * silu(z[i])
// 面向 GatedDeltaNet 类线性注意力层的融合 gated RMSNorm。
// 单 kernel 取代对递推输出与 z 门控支路的独立归一化和门控步骤：
//   variance = 最后一维上的 mean(x^2)
//   out[i] = weight[i] * x[i] * rsqrt(variance + eps) * silu(z[i])

#[derive(Clone)]
pub struct RMSGatedZipMap<T> {
    ptr1: ConstPtr<T>,       // Recurrent output x: [rows, last_dim].
    ptr2: ConstPtr<T>,       // Gate z: [rows, last_dim].
    weight_ptr: ConstPtr<T>, // Norm weight: [last_dim].
    output_ptr: MutPtr<T>,   // Output: [rows, last_dim].

    last_dim: usize, // head_v_dim
    eps: T,
    _marker: PhantomData<T>,
}

impl<T> RMSGatedZipMap<T>
where
    T: Copy + Sqrt + Sigmoid,
{
    pub fn new(
        ptr1: *const T,       // x[rows, last_dim]
        ptr2: *const T,       // z[rows, last_dim]
        weight_ptr: *const T, // weight[last_dim]
        output_ptr: *mut T,   // out[rows, last_dim]
        last_dim: usize,
        eps: T,
    ) -> Self {
        Self {
            ptr1: ConstPtr { ptr: ptr1 },
            ptr2: ConstPtr { ptr: ptr2 },
            weight_ptr: ConstPtr { ptr: weight_ptr },
            output_ptr: MutPtr { ptr: output_ptr },
            last_dim,
            eps,
            _marker: PhantomData,
        }
    }

    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        _total_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        let active_rows = if prefill_size == 0 {
            decode_size
        } else {
            prefill_size
        };

        if let Some((begin, end)) = assign(active_rows, thread_num, thread_id) {
            for index in begin..end {
                unsafe {
                    let offset = index * self.last_dim;
                    self.compute(
                        self.ptr1.ptr.add(offset),
                        self.ptr2.ptr.add(offset),
                        self.output_ptr.ptr.add(offset),
                    );
                }
            }
        }
    }
}

impl<T> ZipMapTrait<T> for RMSGatedZipMap<T>
where
    T: Copy + Sqrt + Sigmoid,
{
    default fn compute(&self, _input_ptr1: *const T, _input_ptr2: *const T, _output_ptr: *mut T) {
        // TODO: compute logic, filled in later
        // 注意：参考实现在 fp32 下做平方和归约，f16 特化时需先提升再截回。
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_gated_zip_map_construct_and_partition() {
        const BATCH_SIZE: usize = 10;
        const LAST_DIM: usize = 18;

        let input_data: Vec<f32> = (0..=17).cycle().take(180).map(|x| x as f32).collect();
        let gate_data = vec![1.0f32; BATCH_SIZE * LAST_DIM];
        let weight = vec![1.0f32; LAST_DIM];
        let mut output_data = vec![0.0f32; BATCH_SIZE * LAST_DIM];

        let operator = RMSGatedZipMap::new(
            input_data.as_ptr(),
            gate_data.as_ptr(),
            weight.as_ptr(),
            output_data.as_mut_ptr(),
            LAST_DIM,
            1e-6,
        );

        // While compute is empty: only assert no panic / partition coverage.
        // compute 为空期间：只验证不 panic、分区覆盖完整。
        let thread_num = 4;
        for thread_id in 0..thread_num {
            operator.run(BATCH_SIZE, 0, BATCH_SIZE, thread_num, thread_id);
        }
        assert!(output_data.iter().all(|&value| value == 0.0));
    }
}
