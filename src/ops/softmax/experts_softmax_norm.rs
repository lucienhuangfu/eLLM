use std::f16;
use std::ops::{AddAssign, Sub};

use crate::ops::traits::map_trait::SoftmaxTrait;
use crate::ops::assign::assign;
use crate::common::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel::scalar;
use crate::num_traits::{exp::Exp, sqrt::Sqrt};
use crate::kernel::x86_64;
use crate::mem_mgr::allocator::allocate_init;

#[derive(Clone)]
pub struct ExpertsSoftmaxNorm<T> {
    // [prefill_size, num_experts]
    ptr1: ConstPtr<T>,
    topk_values_ptr: MutPtr<T>,
    topk_indices_ptr: MutPtr<usize>,
    // Expert routing information
    experts_indicator: MutPtr<bool>,
    indice_ptr: MutPtr<bool>,
    weight_ptr: MutPtr<T>,
    batch_size: usize,
    num_experts: usize,
    num_topk: usize,
    decode_only_flag: bool,
}

impl<T: Sqrt + Default> ExpertsSoftmaxNorm<T> {
    pub fn new(
        ptr1: *const T,
        experts_indicator: *mut bool,
        indice_ptr: *mut bool,
        weight_ptr: *mut T,
        topk_indices_ptr: *mut usize,
        batch_size: usize,
        num_experts: usize,
        num_topk: usize,
        decode_only_flag: bool,
    ) -> Self {
        let length = (batch_size * num_topk);
        Self {
            ptr1: ConstPtr { ptr: ptr1 },

            topk_values_ptr: MutPtr {
                ptr: unsafe { allocate_init::<T>(length, T::default()) },
            },
            topk_indices_ptr: MutPtr {
                ptr: topk_indices_ptr,
            },

            experts_indicator: MutPtr {
                ptr: experts_indicator,
            },
            indice_ptr: MutPtr { ptr: indice_ptr },
            weight_ptr: MutPtr { ptr: weight_ptr },
            // num_tokens: batch_size,
            batch_size,
            num_experts,
            num_topk,
            decode_only_flag,
        }
    }
}

impl<T: Sqrt + Exp + Default + AddAssign + Sub<Output = T> + Copy> ExpertsSoftmaxNorm<T> {
    pub fn run(&self, prefill_size: usize, decode_size: usize, thread_num: usize, thread_id: usize) {
        let task_size = if self.decode_only_flag == true {
            decode_size
        } else {
            prefill_size
        };

        if let Some((begin, end)) = assign(task_size, thread_num, thread_id) {
            let ptr1 = self.ptr1.ptr;
            let topk_indices_ptr = self.topk_indices_ptr.ptr;
            let topk_values_ptr = self.topk_values_ptr.ptr;

            for i in begin..end {
                unsafe {
                    let p1 = i * (self.num_experts);
                    let p2 = i * (self.num_topk);
                    let token_index = i;
                    self.compute(
                        ptr1.add(p1),
                        topk_values_ptr.add(p2),
                        topk_indices_ptr.add(p2),
                        self.experts_indicator.ptr,
                        self.indice_ptr.ptr,
                        self.weight_ptr.ptr,
                        token_index,
                        self.num_experts,
                        self.num_topk,
                    );
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
        scalar::experts_topk_softmax_norm::experts_topk_softmax_norm(
            input_ptr,
            topk_values_ptr,
            topk_indices_ptr,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            token_index,
            self.batch_size,
            input_length,
            output_length,
            true,
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
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        x86_64::f16_512::experts_topk_softmax_norm::experts_topk_softmax_norm(
            input_ptr,
            topk_values_ptr,
            topk_indices_ptr,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            token_index,
            self.batch_size,
            input_length,
            output_length,
            true,
        );
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
            self.batch_size,
            input_length,
            output_length,
            true,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_experts_softmax_norm_f32() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping test.");
            return;
        }

        let batch_size = 2;
        let num_experts = 16;
        let num_topk = 4;
        let num_tokens = batch_size;

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
        let mut indice_ptr = vec![false; (num_experts * num_tokens)];
        let mut weight_ptr = vec![0.0f32; (num_experts * num_tokens)];

        let mut topk_indices_ptr = vec![0; (num_topk * num_tokens)];

        let operator = ExpertsSoftmaxNorm::<f32>::new(
            input_data.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indice_ptr.as_mut_ptr(),
            weight_ptr.as_mut_ptr(),
            topk_indices_ptr.as_mut_ptr(),
            batch_size,
            num_experts,
            num_topk,
            false,
        );

        let thread_num = 1;
        let thread_id = 0;
        operator.run(batch_size, 0, thread_num, thread_id);

        // Verification for token 0
        let mut expected1: Vec<(usize, f32)> = input_data1.iter().copied().enumerate().collect();
        expected1.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_val1 = input_data1
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Calculate denominator using all values (standard softmax)
        let denom1: f32 = input_data1.iter().map(|v| (v - max_val1).exp()).sum();

        // Calculate sum of probabilities for top-k
        let mut prob_sum1 = 0.0;
        for i in 0..num_topk {
            let (_, val) = expected1[i];
            prob_sum1 += (val - max_val1).exp() / denom1;
        }

        for i in 0..num_topk {
            let (idx, val) = expected1[i];
            let prob = (val - max_val1).exp() / denom1;
            let normalized_prob = prob / prob_sum1;

            assert!(experts_indicator[idx]);
            let offset = idx * (num_tokens) + 0;
            assert!(indice_ptr[offset]);
            assert_ulps_eq!(weight_ptr[offset], normalized_prob, epsilon = 1e-4);
        }

        // Verification for token 1
        let mut expected2: Vec<(usize, f32)> = input_data2.iter().copied().enumerate().collect();
        expected2.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_val2 = input_data2
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Calculate denominator using all values (standard softmax)
        let denom2: f32 = input_data2.iter().map(|v| (v - max_val2).exp()).sum();

        // Calculate sum of probabilities for top-k
        let mut prob_sum2 = 0.0;
        for i in 0..num_topk {
            let (_, val) = expected2[i];
            prob_sum2 += (val - max_val2).exp() / denom2;
        }

        for i in 0..num_topk {
            let (idx, val) = expected2[i];
            let prob = (val - max_val2).exp() / denom2;
            let normalized_prob = prob / prob_sum2;

            assert!(experts_indicator[idx]);
            let offset = idx * (num_tokens) + 1;
            assert!(indice_ptr[offset]);
            assert_ulps_eq!(weight_ptr[offset], normalized_prob, epsilon = 1e-4);
        }
    }

    #[test]
    // #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    fn test_experts_softmax_norm_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let sequence_chunk_size = 1;
        let batch_size = 6;
        let num_experts = 128;
        let num_topk = 8;
        let num_tokens = sequence_chunk_size * batch_size;

        // Generate input data for batch_size tokens * num_experts
        let mut input_vals: Vec<f32> = Vec::with_capacity((num_tokens * num_experts));
        for t in 0..num_tokens {
            for i in 0..num_experts {
                // Create some variation based on token index t and expert index i
                let v = ((i as f32 + t as f32 * 13.0) * 1.1) % 20.0 - 10.0;
                input_vals.push(v);
            }
            // Ensure distinct top values for this token to avoid sorting ambiguity in tests
            let base = t * num_experts;
            // Make index `t` the absolute winner to distinguish tokens slightly
            input_vals[(base + (t % num_experts))] = 30.0;
            // Make index `(t+1)` second
            input_vals[(base + ((t + 1) % num_experts))] = 25.0;
        }

        let input_data: Vec<f16> = input_vals.iter().map(|&x| (x as f16)).collect();

        let mut experts_indicator = vec![false; num_experts];
        let mut indice_ptr = vec![false; (num_experts * num_tokens)];
        let mut weight_ptr = vec![(0.0f16); (num_experts * num_tokens)];
        let mut topk_indices_ptr = vec![0; (num_topk * num_tokens)];

        let operator = ExpertsSoftmaxNorm::<f16>::new(
            input_data.as_ptr(),
            experts_indicator.as_mut_ptr(),
            indice_ptr.as_mut_ptr(),
            weight_ptr.as_mut_ptr(),
            topk_indices_ptr.as_mut_ptr(),
            batch_size,
            num_experts,
            num_topk,
            false,
        );

        let thread_num = 8;
        for thread_id in 0..thread_num {
            operator.run(batch_size, 0, thread_num, thread_id);
        }

        // Verification
        for t in 0..num_tokens {
            let start = (t * num_experts);
            let end = start + (num_experts);
            let token_input = &input_vals[start..end];

            let mut expected: Vec<(usize, f32)> = token_input.iter().copied().enumerate().collect();
            // Sort descending by value
            expected.sort_by(|a, b| b.1.total_cmp(&a.1));

            let max_val = token_input
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = token_input.iter().map(|v| (v - max_val).exp()).sum();

            for i in 0..num_topk {
                let (idx, val) = expected[i];
                let prob = (val - max_val).exp() / denom;

                assert!(
                    experts_indicator[idx],
                    "Expert {} should be selected (token {})",
                    idx, t
                );

                let offset = idx * (num_tokens) + (t);
                assert!(
                    indice_ptr[offset],
                    "Index ptr for expert {} token {} should be true",
                    idx, t
                );

                let prob_f16 = (prob as f16);
                let weight = weight_ptr[offset];

                assert!(
                    (weight - prob_f16).abs() < (1e-3f16),
                    "Weight mismatch for expert {} token {}: expected {:?}, got {:?}",
                    idx,
                    t,
                    prob_f16,
                    weight
                );
            }
        }
    }
}






