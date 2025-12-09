use std::f16;
use std::ops::{AddAssign, Sub};
use std::ptr;

use super::map_trait::TopKSoftmaxTrait;
use crate::compiler::assign::assign;
use crate::init::record::{BatchRecord, Phase, TokenRecord};
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::kernel::generic::exp::Exp;
// use crate::kernel::generic::from_usize::FromUsize;
use crate::kernel::generic::sqrt::Sqrt;

// use crate::memory::allocator::allocate_init;

#[derive(Clone)]
pub struct TopKSoftmax<T> {
    input_indices_ptr: ConstPtr<usize>,
    input_values_ptr: ConstPtr<T>,
    sums_ptr: ConstPtr<T>,
    token_ptr: ConstPtr<TokenRecord>,
    user_ptr: MutPtr<BatchRecord>,
    output_indices_ptr: MutPtr<usize>,
    output_values_ptr: MutPtr<T>,
    output_sequences: MutPtr<usize>,
    batch_size: usize,
    topk_size: usize,
    eos_id: usize,
}
impl<T: Sqrt + Exp + Default + AddAssign + Sub<Output = T> + Copy> TopKSoftmax<T> {
    pub fn new(
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        sums_ptr: *const T,
        token_ptr: *const TokenRecord,
        user_ptr: *mut BatchRecord,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        output_sequences: *mut usize,
        batch_size: usize,
        topk_size: usize,
        eos_id: usize,
    ) -> Self {
        Self {
            input_indices_ptr: ConstPtr {
                ptr: input_indices_ptr,
            },
            input_values_ptr: ConstPtr {
                ptr: input_values_ptr,
            },
            sums_ptr: ConstPtr { ptr: sums_ptr },
            token_ptr: ConstPtr { ptr: token_ptr },
            user_ptr: MutPtr { ptr: user_ptr },
            output_indices_ptr: MutPtr {
                ptr: output_indices_ptr,
            },
            output_values_ptr: MutPtr {
                ptr: output_values_ptr,
            },
            output_sequences: MutPtr {
                ptr: output_sequences,
            },
            batch_size,
            topk_size,
            eos_id,
        }
    }

    pub fn run(&self, token_size: usize, decode_size: usize, thread_num: usize, thread_id: usize) {
        if let Some((begin, end)) = assign(decode_size, thread_num, thread_id) {
            let mut input_indices_ptr = self.input_indices_ptr.ptr;
            let mut input_values_ptr = self.input_values_ptr.ptr;
            let mut sums_ptr = self.sums_ptr.ptr;
            let mut token_ptr = self.token_ptr.ptr;
            let mut output_indices_ptr = self.output_indices_ptr.ptr;
            let mut output_values_ptr = self.output_values_ptr.ptr;
            let mut output_sequences_ptr = self.output_sequences.ptr;

            for i in begin..end {
                unsafe {
                    let batch_index = (*token_ptr.add(i)).batch_index;
                    let position_index = (*token_ptr.add(i)).position_index;
                    let input_stride = batch_index * self.topk_size * thread_num;
                    let output_stride = batch_index * self.topk_size;
                    let token_ptr = output_sequences_ptr
                        .add((position_index + 1) * self.batch_size + batch_index);
                    let _output_indices_ptr = output_indices_ptr.add(output_stride);
                    self.compute(
                        input_indices_ptr.add(input_stride),
                        input_values_ptr.add(input_stride),
                        sums_ptr.add(batch_index),
                        _output_indices_ptr,
                        output_values_ptr.add(output_stride),
                        // token_ptr,
                        thread_num,
                        self.topk_size,
                    );
                    let predict_token = *_output_indices_ptr;
                    ptr::write(token_ptr, predict_token);
                    if predict_token == self.eos_id {
                        let user_record = self.user_ptr.ptr.add(batch_index);
                        (*user_record).phase = Phase::Eos;
                    }
                }
            }
        }
    }
}
impl<T: Sqrt + Exp + Default + AddAssign + Sub<Output = T> + Copy> TopKSoftmaxTrait<T>
    for TopKSoftmax<T>
{
    default fn compute(
        &self,
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        sums_ptr: *const T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        // output_token_ptr: *mut usize,
        thread_num: usize,
        topk_size: usize,
    ) {
        kernel::generic::truncated_topk_softmax::truncated_topk_softmax(
            input_values_ptr,
            input_indices_ptr,
            // sums_ptr,
            output_values_ptr,
            output_indices_ptr,
            // output_token_ptr,
            thread_num,
            topk_size,
        );
    }
}

impl TopKSoftmaxTrait<f16> for TopKSoftmax<f16> {
    fn compute(
        &self,
        input_indices_ptr: *const usize,
        input_values_ptr: *const f16,
        sums_ptr: *const f16,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut f16,
        // output_token_ptr: *mut usize,
        thread_num: usize,
        topk_size: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::truncated_topk_softmax::truncated_topk_softmax(
            input_values_ptr,
            input_indices_ptr,
            // sums_ptr,
            output_values_ptr,
            output_indices_ptr,
            // output_token_ptr,
            thread_num,
            topk_size,
        );
        /*
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
    fn compute(
        &self,
        input_indices_ptr: *const usize,
        input_values_ptr: *const f32,
        sums_ptr: *const f32,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut f32,
        // output_token_ptr: *mut usize,
        thread_num: usize,
        topk_size: usize,
    ) {
        kernel::x86_64::f32_256::truncated_topk_softmax::truncated_topk_softmax(
            input_values_ptr,
            input_indices_ptr,
            sums_ptr,
            output_values_ptr,
            output_indices_ptr,
            // output_token_ptr,
            thread_num,
            topk_size,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn test_topk_softmax_f32() {
        let sequence_length = 2;
        let batch_size = 2;
        let topk_size = 8;
        let thread_num = 4;
        let eos_id = 100;

        let total_candidates_per_item = topk_size * thread_num;
        let input_len = batch_size * total_candidates_per_item;

        let mut input_values = Vec::<f32>::with_capacity(input_len);
        let mut input_indices = Vec::<usize>::with_capacity(input_len);
        let mut token_records = Vec::with_capacity(batch_size);
        let mut user_records = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            token_records.push(TokenRecord {
                token_id: 0,
                batch_index: i,
                position_index: 0,
            });
            user_records.push(BatchRecord {
                sequence_index: i,
                // snapshot_sequence_index: 0,
                kv_index: 0,
                phase: Phase::Decode,
            });
            for j in 0..total_candidates_per_item {
                // Create some decreasing values for each batch item
                input_values.push(5.0 - (j as f32 * 0.1) - (i as f32));
                input_indices.push(i * 1000 + j); // Unique indices
            }
        }

        let sums = vec![0.0f32; batch_size];
        let mut output_values = vec![0.0f32; batch_size * topk_size];
        let mut output_indices = vec![0usize; batch_size * topk_size];
        let mut output_sequences = vec![0usize; batch_size * sequence_length];

        let operator = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            sums.as_ptr(),
            token_records.as_ptr(),
            user_records.as_mut_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_size,
            topk_size,
            eos_id,
        );

        for i in 0..thread_num {
            operator.run(batch_size, batch_size, thread_num, i);
        }

        // Verification
        for i in 0..batch_size {
            let item_input_values =
                &input_values[i * total_candidates_per_item..(i + 1) * total_candidates_per_item];
            let item_input_indices =
                &input_indices[i * total_candidates_per_item..(i + 1) * total_candidates_per_item];

            let mut paired: Vec<_> = item_input_values
                .iter()
                .copied()
                .zip(item_input_indices.iter().copied())
                .collect();
            paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let topk = &paired[..topk_size];
            let max_val = topk[0].0;
            let denom: f32 = topk.iter().map(|(v, _)| (v - max_val).exp()).sum();

            let expected_probs: Vec<f32> = topk
                .iter()
                .map(|(v, _)| (v - max_val).exp() / denom)
                .collect();
            let expected_indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();

            let output_vals_slice = &output_values[i * topk_size..(i + 1) * topk_size];
            let output_idx_slice = &output_indices[i * topk_size..(i + 1) * topk_size];

            assert_ulps_eq!(output_vals_slice, expected_probs.as_slice(), max_ulps = 4);
            assert_eq!(output_idx_slice, expected_indices.as_slice());
            assert_eq!(output_sequences[batch_size + i], expected_indices[0]);
        }
    }

    #[test]
    fn test_topk_softmax_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let sequence_length = 2;
        let batch_size = 2;
        let topk_size = 8;
        let thread_num = 4;
        let eos_id = 100;

        let total_candidates_per_item = topk_size * thread_num;
        let input_len = batch_size * total_candidates_per_item;

        let mut input_values = Vec::<f16>::with_capacity(input_len);
        let mut input_indices = Vec::<usize>::with_capacity(input_len);
        let mut token_records = Vec::with_capacity(batch_size);
        let mut user_records = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            token_records.push(TokenRecord {
                token_id: 0,
                batch_index: i,
                position_index: 0,
            });
            user_records.push(BatchRecord {
                sequence_index: i,
                kv_index: 0,
                phase: Phase::Decode,
            });
            for j in 0..total_candidates_per_item {
                // Create some decreasing values for each batch item
                let val = 5.0 - (j as f32 * 0.1) - (i as f32);
                input_values.push(val as f16);
                input_indices.push(i * 1000 + j); // Unique indices
            }
        }

        let sums = vec![0.0 as f16; batch_size];
        let mut output_values = vec![0.0 as f16; batch_size * topk_size];
        let mut output_indices = vec![0usize; batch_size * topk_size];
        let mut output_sequences = vec![0usize; batch_size * sequence_length];

        let operator = TopKSoftmax::<f16>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            sums.as_ptr(),
            token_records.as_ptr(),
            user_records.as_mut_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_size,
            topk_size,
            eos_id,
        );

        for i in 0..thread_num {
            operator.run(batch_size, batch_size, thread_num, i);
        }

        // Verification
        for i in 0..batch_size {
            let item_input_values =
                &input_values[i * total_candidates_per_item..(i + 1) * total_candidates_per_item];
            let item_input_indices =
                &input_indices[i * total_candidates_per_item..(i + 1) * total_candidates_per_item];

            let mut paired: Vec<_> = item_input_values
                .iter()
                .copied()
                .zip(item_input_indices.iter().copied())
                .collect();
            paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let topk = &paired[..topk_size];
            let max_val = topk[0].0 as f32;

            let topk_f32: Vec<(f32, usize)> =
                topk.iter().map(|(v, idx)| (*v as f32, *idx)).collect();
            let denom: f32 = topk_f32.iter().map(|(v, _)| (v - max_val).exp()).sum();

            let expected_probs: Vec<f32> = topk_f32
                .iter()
                .map(|(v, _)| (v - max_val).exp() / denom)
                .collect();
            let expected_indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();

            let output_vals_slice = &output_values[i * topk_size..(i + 1) * topk_size];
            let output_idx_slice = &output_indices[i * topk_size..(i + 1) * topk_size];

            for k in 0..topk_size {
                let out_val = (output_vals_slice[k] as f32);
                let expected = expected_probs[k];
                assert!(
                    (out_val - expected).abs() < 1e-3,
                    "Mismatch at batch {} index {}: got {}, expected {}",
                    i,
                    k,
                    out_val,
                    expected
                );
                assert_eq!(output_idx_slice[k], expected_indices[k]);
            }
            assert_eq!(output_sequences[batch_size + i], expected_indices[0]);
        }
    }
}
