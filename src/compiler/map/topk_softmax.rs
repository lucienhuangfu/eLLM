use std::f16;
use std::ops::{AddAssign, Sub};
use std::ptr;

use super::map_trait::TopKSoftmaxTrait;
use crate::init::record::{BatchList, Phase, TaskList};
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
    batch_list_ptr: MutPtr<BatchList>,
    decode_list_ptr: ConstPtr<TaskList>,
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
        batch_list_ptr: *mut BatchList,
        decode_list_ptr: *const TaskList,
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
            batch_list_ptr: MutPtr { ptr: batch_list_ptr },
            decode_list_ptr: ConstPtr { ptr: decode_list_ptr },
            batch_size,
            topk_size,
            eos_id,
        }
    }

    pub fn run(
        &self,
        _prefill_size: usize,
        decode_size: usize,
        thread_num: usize,
        thread_id: usize,
    ) {
        if decode_size == 0 {
            return;
        }

        let decode_list_ptr = self.decode_list_ptr.ptr;
        let batch_list_ptr = self.batch_list_ptr.ptr;
        if decode_list_ptr.is_null() || batch_list_ptr.is_null() {
            return;
        }

        unsafe {
            let decode_list = &*decode_list_ptr;
            if thread_id >= decode_list.tasks.len() {
                return;
            }

            let input_indices_ptr = self.input_indices_ptr.ptr;
            let input_values_ptr = self.input_values_ptr.ptr;
            let output_indices_ptr = self.output_indices_ptr.ptr;
            let output_values_ptr = self.output_values_ptr.ptr;
            let output_sequences_ptr = self.output_sequences.ptr;

            let batch_list = &mut *batch_list_ptr;

            let task = &decode_list.tasks[thread_id];
            for i in 0..task.current_size {
                let slice = &task.slices[i];
                let batch_index = slice.batch_index;
                let mut sequence_index = slice.sequence_index;
                let token_start_index = slice.token_start_index;

                for t in 0..slice.length {
                    let token_index = token_start_index + t;
                    let input_stride = token_index * self.topk_size * thread_num;
                    let output_stride = token_index * self.topk_size;

                    self.compute(
                        input_indices_ptr.add(input_stride),
                        input_values_ptr.add(input_stride),
                        output_indices_ptr.add(output_stride),
                        output_values_ptr.add(output_stride),
                        thread_num,
                        self.topk_size,
                    );

                    let predict_token = *output_indices_ptr.add(output_stride);
                    let out_offset = sequence_index * self.batch_size + batch_index;
                    ptr::write(output_sequences_ptr.add(out_offset), predict_token);
                    sequence_index = sequence_index.saturating_add(1);

                    if batch_index < batch_list.current_size {
                        let record = &mut batch_list.records[batch_index];
                        if predict_token == self.eos_id {
                            record.phase = Phase::Eos;
                            record.notify.notify_one();
                        }
                    }
                }

                if batch_index < batch_list.current_size {
                    batch_list.records[batch_index].sequence_index = sequence_index;
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
    use crate::init::record::{
        BatchList, BatchRecord, Phase, SequenceSlice, TaskList, ThreadTask
    };
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
                sequence_index: 1,
                snapshot_sequence_index: 0,
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

        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut tasks = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for batch_index in start..end {
                slices.push(SequenceSlice {
                    batch_index,
                    sequence_index: 1,
                    token_start_index: batch_index,
                    length: 1,
                });
            }
            tasks.push(ThreadTask {
                slices,
                current_size: end.saturating_sub(start),
            });
        }
        let decode_list = TaskList {
            tasks,
            current_size: thread_num,
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
            &mut batch_list as *mut BatchList,
            &decode_list as *const TaskList,
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
                sequence_index: 1,
                snapshot_sequence_index: 0,
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

        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut tasks = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for batch_index in start..end {
                slices.push(SequenceSlice {
                    batch_index,
                    sequence_index: 1,
                    token_start_index: batch_index,
                    length: 1,
                });
            }
            tasks.push(ThreadTask {
                slices,
                current_size: end.saturating_sub(start),
            });
        }
        let decode_list = TaskList {
            tasks,
            current_size: thread_num,
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
            &mut batch_list as *mut BatchList,
            &decode_list as *const TaskList,
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

