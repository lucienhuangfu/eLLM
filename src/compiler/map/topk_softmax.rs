use std::f16;
use std::ops::{AddAssign, Sub};
use std::ptr;

use super::map_trait::TopKSoftmaxTrait;
use crate::compiler::assign::assign;
use crate::init::record::{BatchList, BatchRecord, Phase, TokenList, TokenRecord};
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};
use crate::kernel;
use crate::kernel::generic::exp::Exp;
use crate::kernel::generic::sqrt::Sqrt;

#[derive(Clone)]
pub struct TopKSoftmax<T> {
    input_indices_ptr: ConstPtr<usize>,
    input_values_ptr: ConstPtr<T>,
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
            let mut input_indices_ptr = self.input_indices_ptr.ptr;
            let mut input_values_ptr = self.input_values_ptr.ptr;
            let mut output_indices_ptr = self.output_indices_ptr.ptr;
            let mut output_values_ptr = self.output_values_ptr.ptr;

            let mut output_sequences_ptr = self.output_sequences.ptr;

            for _ in begin..end {
                let index = row_index * self.batch_size + col_index;
                unsafe {
                    let input_stride = index * self.topk_size * thread_num;
                    let output_stride = index * self.topk_size;
                    let token_index = index + position_begin * self.batch_size;
                    let token_ptr = output_sequences_ptr.add(token_index);
                    self.compute(
                        input_indices_ptr.add(input_stride),
                        input_values_ptr.add(input_stride),
                        output_indices_ptr.add(output_stride),
                        output_values_ptr.add(output_stride),
                        token_ptr,
                        thread_num,
                        self.topk_size,
                    );
                }
            }

            // --- Phase 2: Process Lift Tokens ---
            let lift_loop_begin = std::cmp::max(begin, decode_end_index);

            if lift_loop_begin < end {
                unsafe {
                    let lift_records_ptr = (*self.token_ptr.ptr).lift_records.as_ptr();
                    // Calculate offset into lift_records
                    let lift_start_offset = lift_loop_begin - decode_end_index;
                    let lift_count = end - lift_loop_begin;

                    for i in 0..lift_count {
                        let lift_record = &*lift_records_ptr.add(lift_start_offset + i);
                        let batch_index = lift_record.prefill_end_index;
                        // Double dereference is unavoidable here without changing data layout
                        let position_index = (*token_ptr_base.add(batch_index)).position_index;

                        self.process_one(
                            batch_index,
                            position_index,
                            thread_num,
                            input_stride_factor,
                            topk_size,
                            batch_size,
                            eos_id,
                            input_indices_ptr,
                            input_values_ptr,
                            sums_ptr,
                            output_indices_ptr,
                            output_values_ptr,
                            output_sequences_ptr,
                            batch_ptr_base,
                        );
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn process_one(
        &self,
        batch_index: usize,
        position_index: usize,
        thread_num: usize,
        input_stride_factor: usize,
        topk_size: usize,
        batch_size: usize,
        eos_id: usize,
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        sums_ptr: *const T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        output_sequences_ptr: *mut usize,
        batch_ptr: *mut BatchRecord,
    ) {
        // Optimized stride calculation
        let input_stride = batch_index * input_stride_factor;
        let output_stride = batch_index * topk_size;

        let token_ptr = output_sequences_ptr.add((position_index + 1) * batch_size + batch_index);
        let _output_indices_ptr = output_indices_ptr.add(output_stride);

        self.compute(
            input_indices_ptr.add(input_stride),
            input_values_ptr.add(input_stride),
            sums_ptr.add(batch_index),
            _output_indices_ptr,
            output_values_ptr.add(output_stride),
            thread_num,
            topk_size,
        );

        let predict_token = *_output_indices_ptr;
        ptr::write(token_ptr, predict_token);

        if predict_token == eos_id {
            let batch_record = batch_ptr.add(batch_index);
            (*batch_record).phase = Phase::Eos;
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
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut f32,
        // output_token_ptr: *mut usize,
        thread_num: usize,
        topk_size: usize,
    ) {
        kernel::x86_64::f32_256::truncated_topk_softmax::truncated_topk_softmax(
            input_values_ptr,
            input_indices_ptr,
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
    use crate::init::record::PrefillEndRecord;
    use approx::assert_ulps_eq;

    #[test]
    fn test_topk_softmax_f32() {
        let sequence_length = 2;
        let batch_size = 2;
        let topk_size = 8;
        let thread_num = 4;
        let eos_id = 100;

        let total_candidates_per_item = topk_size * thread_num;
        let input_len = (batch_size * total_candidates_per_item);

        let mut input_values = Vec::<f32>::with_capacity(input_len);
        let mut input_indices = Vec::<usize>::with_capacity(input_len);
        let mut token_records_vec = Vec::with_capacity(batch_size);
        let mut user_records_vec = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            token_records_vec.push(TokenRecord {
                // token_id: 0,
                batch_index: i,
                position_index: 0,
            });
            user_records_vec.push(BatchRecord {
                sequence_index: i,
                // snapshot_sequence_index: 0,
                kv_index: 0,
                phase: Phase::Decode,
                prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            });
            for j in 0..total_candidates_per_item {
                // Create some decreasing values for each batch item
                input_values.push(5.0 - (j as f32 * 0.1) - (i as f32));
                input_indices.push(i * 1000 + j); // Unique indices
            }
        }

        let token_list = TokenList {
            token_records: token_records_vec.into_boxed_slice(),
            current_token_size: batch_size,
            lift_records: Box::new([]),
            current_lift_size: 0,
        };

        let mut batch_list = BatchList {
            records: user_records_vec.into_boxed_slice(),
            current_size: batch_size,
        };

        let sums = vec![0.0f32; batch_size];
        let mut output_values = vec![0.0f32; (batch_size * topk_size)];
        let mut output_indices = vec![0; (batch_size * topk_size)];
        let mut output_sequences = vec![0; (batch_size * sequence_length)];

        let operator = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
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
            let i_usize = i;
            let total_candidates_per_item_usize = total_candidates_per_item;
            let topk_size_usize = topk_size;

            let item_input_values = &input_values[i_usize * total_candidates_per_item_usize
                ..(i_usize + 1) * total_candidates_per_item_usize];
            let item_input_indices = &input_indices[i_usize * total_candidates_per_item_usize
                ..(i_usize + 1) * total_candidates_per_item_usize];

            let mut paired: Vec<_> = item_input_values
                .iter()
                .copied()
                .zip(item_input_indices.iter().copied())
                .collect();
            paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let topk = &paired[..topk_size_usize];
            let max_val = topk[0].0;
            let denom: f32 = topk.iter().map(|(v, _)| (v - max_val).exp()).sum();

            let expected_probs: Vec<f32> = topk
                .iter()
                .map(|(v, _)| (v - max_val).exp() / denom)
                .collect();
            let expected_indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();

            let output_vals_slice =
                &output_values[i_usize * topk_size_usize..(i_usize + 1) * topk_size_usize];
            let output_idx_slice =
                &output_indices[i_usize * topk_size_usize..(i_usize + 1) * topk_size_usize];

            assert_ulps_eq!(output_vals_slice, expected_probs.as_slice(), max_ulps = 4);
            assert_eq!(output_idx_slice, expected_indices.as_slice());
            assert_eq!(
                output_sequences[(batch_size) + i_usize],
                expected_indices[0]
            );
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
        let input_len = (batch_size * total_candidates_per_item);

        let mut input_values = Vec::<f16>::with_capacity(input_len);
        let mut input_indices = Vec::<usize>::with_capacity(input_len);
        let mut token_records_vec = Vec::with_capacity(batch_size);
        let mut user_records_vec = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            token_records_vec.push(TokenRecord {
                // token_id: 0,
                batch_index: i,
                position_index: 0,
            });
            user_records_vec.push(BatchRecord {
                sequence_index: i,
                kv_index: 0,
                phase: Phase::Decode,
                prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            });
            for j in 0..total_candidates_per_item {
                // Create some decreasing values for each batch item
                let val = 5.0 - (j as f32 * 0.1) - (i as f32);
                input_values.push(val as f16);
                input_indices.push(i * 1000 + j); // Unique indices
            }
        }

        let token_list = TokenList {
            token_records: token_records_vec.into_boxed_slice(),
            current_token_size: batch_size,
            lift_records: Box::new([]),
            current_lift_size: 0,
        };

        let mut batch_list = BatchList {
            records: user_records_vec.into_boxed_slice(),
            current_size: batch_size,
        };

        let sums = vec![0.0 as f16; batch_size];
        let mut output_values = vec![0.0 as f16; (batch_size * topk_size)];
        let mut output_indices = vec![0; (batch_size * topk_size)];
        let mut output_sequences = vec![0; (batch_size * sequence_length)];

        let operator = TopKSoftmax::<f16>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
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
            let i_usize = i;
            let total_candidates_per_item_usize = total_candidates_per_item;
            let topk_size_usize = topk_size;

            let item_input_values = &input_values[i_usize * total_candidates_per_item_usize
                ..(i_usize + 1) * total_candidates_per_item_usize];
            let item_input_indices = &input_indices[i_usize * total_candidates_per_item_usize
                ..(i_usize + 1) * total_candidates_per_item_usize];

            let mut paired: Vec<_> = item_input_values
                .iter()
                .copied()
                .zip(item_input_indices.iter().copied())
                .collect();
            paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let topk = &paired[..topk_size_usize];
            let max_val = topk[0].0 as f32;

            let topk_f32: Vec<(f32, usize)> =
                topk.iter().map(|(v, idx)| (*v as f32, *idx)).collect();
            let denom: f32 = topk_f32.iter().map(|(v, _)| (v - max_val).exp()).sum();

            let expected_probs: Vec<f32> = topk_f32
                .iter()
                .map(|(v, _)| (v - max_val).exp() / denom)
                .collect();
            let expected_indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();

            let output_vals_slice =
                &output_values[i_usize * topk_size_usize..(i_usize + 1) * topk_size_usize];
            let output_idx_slice =
                &output_indices[i_usize * topk_size_usize..(i_usize + 1) * topk_size_usize];

            for k in 0..topk_size_usize {
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
            assert_eq!(
                output_sequences[(batch_size) + i_usize],
                expected_indices[0]
            );
        }
    }
}
