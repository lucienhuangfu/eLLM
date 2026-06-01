use std::f16;
use std::ops::{AddAssign, Div, Mul, Sub};
use std::ptr;

use crate::kernel;
use crate::num_traits::Exp;
use crate::num_traits::FromNumber;
use crate::num_traits::Sqrt;
use crate::operators::assign::assign;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::operators::traits::TopKSoftmaxTrait;
use crate::runtime::scheduling::SequenceSlice;
use crate::runtime::{Phase, SequenceState};
use rand::Rng;

#[derive(Clone)]
pub struct TopKSoftmax<T> {
    input_indices_ptr: ConstPtr<usize>,
    input_values_ptr: ConstPtr<T>,
    output_indices_ptr: MutPtr<usize>,
    output_values_ptr: MutPtr<T>,
    output_sequences: MutPtr<usize>,
    batch_temperature: MutPtr<T>,
    sequence_stride: usize,
    input_top_k: usize,
    top_k: usize,
    top_p: T,
    min_p: T,
    do_sample: bool,
    eos_ids: Vec<usize>,
}

impl<
        T: Sqrt
            + Exp
            + Default
            + AddAssign
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + PartialOrd
            + Copy
            + FromNumber,
    > TopKSoftmax<T>
{
    pub fn new(
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        output_sequences: *mut usize,
        batch_temperature: *mut T,
        sequence_stride: usize,
        input_top_k: usize,
        top_k: usize,
        eos_ids: Vec<usize>,
    ) -> Self {
        Self::with_sampling(
            input_indices_ptr,
            input_values_ptr,
            output_indices_ptr,
            output_values_ptr,
            output_sequences,
            batch_temperature,
            sequence_stride,
            input_top_k,
            top_k,
            T::from_f32(1.0),
            T::default(),
            false,
            eos_ids,
        )
    }

    pub fn with_sampling(
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        output_sequences: *mut usize,
        batch_temperature: *mut T,
        sequence_stride: usize,
        input_top_k: usize,
        top_k: usize,
        top_p: T,
        min_p: T,
        do_sample: bool,
        eos_ids: Vec<usize>,
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
            batch_temperature: MutPtr {
                ptr: batch_temperature,
            },
            sequence_stride,
            input_top_k,
            top_k,
            top_p,
            min_p,
            do_sample,
            eos_ids,
        }
    }

    pub fn run(
        &self,
        prefill_size: usize,
        decode_size: usize,
        thread_num: usize,
        thread_id: usize,
        _prefill_list: &[Vec<SequenceSlice>],
        decode_list: &[SequenceSlice],
        batch_list: &mut Vec<SequenceState>,
    ) {
        if prefill_size == 0 && decode_size == 0 {
            return;
        }

        let Some((begin, end)) = assign(decode_list.len(), thread_num, thread_id) else {
            return;
        };

        unsafe {
            let input_indices_ptr = self.input_indices_ptr.ptr;
            let input_values_ptr = self.input_values_ptr.ptr;
            let output_indices_ptr = self.output_indices_ptr.ptr;
            let output_values_ptr = self.output_values_ptr.ptr;
            let output_sequences_ptr = self.output_sequences.ptr;

            for (row_index, slice) in decode_list.iter().enumerate().take(end).skip(begin) {
                let batch_index = slice.batch_index;
                let slice_length = slice.length;

                if slice_length == 0 || batch_index >= batch_list.len() {
                    continue;
                }

                let record = &mut batch_list[batch_index];
                self.update_prefill_state(record, slice_length);

                if !slice.last_token_flag || !matches!(record.phase, Phase::Decode) {
                    continue;
                }

                let write_sequence_index = record.kv_index;
                if write_sequence_index >= self.sequence_stride {
                    record.phase = Phase::Eos;
                    record.notify.notify_one();
                    continue;
                }

                let temperature = self.get_temperature(batch_index);
                let input_stride = row_index * self.input_top_k * thread_num;
                let output_stride = row_index * self.top_k;

                self.compute(
                    input_indices_ptr.add(input_stride),
                    input_values_ptr.add(input_stride),
                    temperature,
                    output_indices_ptr.add(output_stride),
                    output_values_ptr.add(output_stride),
                    thread_num,
                    self.input_top_k,
                    self.top_k,
                );

                let predict_token = self.filter_and_sample(
                    output_indices_ptr.add(output_stride),
                    output_values_ptr.add(output_stride),
                    self.top_k,
                );

                let out_offset = batch_index * self.sequence_stride + write_sequence_index;
                ptr::write(output_sequences_ptr.add(out_offset), predict_token);

                record.sequence_index = write_sequence_index;
                record.kv_index = record.kv_index.saturating_add(1);

                if self.eos_ids.contains(&predict_token) {
                    record.phase = Phase::Eos;
                }
                // Notify after every decoded token (including EOS) so that
                // streaming handlers can read the token immediately.
                record.notify.notify_one();
            }
        }
    }

    fn update_prefill_state(&self, record: &mut SequenceState, slice_length: usize) {
        if matches!(record.phase, Phase::Prefill) {
            record.sequence_index = record.sequence_index.saturating_add(slice_length);
            record.kv_index = record.kv_index.saturating_add(slice_length);
            record.filling_length = record.filling_length.saturating_sub(slice_length);

            if record.filling_length == 0 {
                record.phase = Phase::Decode;
            }
        }
    }

    fn get_temperature(&self, batch_index: usize) -> T {
        let temperature = unsafe { *self.batch_temperature.ptr.add(batch_index) };
        if temperature <= T::default() {
            T::from_f32(1.0)
        } else {
            temperature
        }
    }

    unsafe fn filter_and_sample(
        &self,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        len: usize,
    ) -> usize {
        let zero = T::default();
        let one = T::from_f32(1.0);
        let top_p_enabled = self.top_p > zero && self.top_p < one;
        let min_p_enabled = self.min_p > zero;

        let max_prob = *output_values_ptr;
        let min_prob_threshold = if min_p_enabled {
            max_prob * self.min_p
        } else {
            zero
        };

        let kept_mass = self.apply_min_p_filter(output_values_ptr, len, min_prob_threshold);

        if kept_mass <= zero {
            return self.handle_zero_mass(output_indices_ptr, output_values_ptr, len);
        }

        let cutoff = if top_p_enabled {
            self.compute_top_p_cutoff(output_values_ptr, len, kept_mass)
        } else {
            len
        };

        let selected_mass = self.sum_selected_probs(output_values_ptr, cutoff);

        if selected_mass <= zero {
            return self.handle_zero_mass(output_indices_ptr, output_values_ptr, len);
        }

        self.normalize_probs(output_values_ptr, len, cutoff, selected_mass);

        if !self.do_sample {
            return *output_indices_ptr;
        }

        self.perform_sampling(output_indices_ptr, output_values_ptr, cutoff)
    }

    unsafe fn apply_min_p_filter(
        &self,
        output_values_ptr: *mut T,
        len: usize,
        min_prob_threshold: T,
    ) -> T {
        let zero = T::default();
        let mut kept_mass = zero;

        for i in 0..len {
            let prob = *output_values_ptr.add(i);
            if prob >= min_prob_threshold {
                kept_mass += prob;
            } else {
                ptr::write(output_values_ptr.add(i), zero);
            }
        }

        kept_mass
    }

    unsafe fn handle_zero_mass(
        &self,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        len: usize,
    ) -> usize {
        let zero = T::default();
        let one = T::from_f32(1.0);

        ptr::write(output_values_ptr, one);
        for i in 1..len {
            ptr::write(output_values_ptr.add(i), zero);
        }

        *output_indices_ptr
    }

    unsafe fn compute_top_p_cutoff(
        &self,
        output_values_ptr: *mut T,
        len: usize,
        kept_mass: T,
    ) -> usize {
        let zero = T::default();
        let target_mass = kept_mass * self.top_p;
        let mut cumulative = zero;

        for i in 0..len {
            let prob = *output_values_ptr.add(i);
            if prob <= zero {
                continue;
            }

            cumulative += prob;
            if cumulative >= target_mass {
                return i + 1;
            }
        }

        len.max(1)
    }

    unsafe fn sum_selected_probs(&self, output_values_ptr: *mut T, cutoff: usize) -> T {
        let mut selected_mass = T::default();
        for i in 0..cutoff {
            selected_mass += *output_values_ptr.add(i);
        }
        selected_mass
    }

    unsafe fn normalize_probs(
        &self,
        output_values_ptr: *mut T,
        len: usize,
        cutoff: usize,
        selected_mass: T,
    ) {
        let zero = T::default();
        let inv_mass = T::from_f32(1.0) / selected_mass;

        for i in 0..len {
            let prob = if i < cutoff {
                *output_values_ptr.add(i) * inv_mass
            } else {
                zero
            };
            ptr::write(output_values_ptr.add(i), prob);
        }
    }

    unsafe fn perform_sampling(
        &self,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        cutoff: usize,
    ) -> usize {
        let zero = T::default();
        let mut rng = rand::thread_rng();
        let sample = T::from_f32(rng.gen::<f32>());
        let mut cumulative = zero;

        for i in 0..cutoff {
            cumulative += *output_values_ptr.add(i);
            if sample <= cumulative || i + 1 == cutoff {
                return *output_indices_ptr.add(i);
            }
        }

        *output_indices_ptr
    }
}

impl<
        T: Sqrt
            + Exp
            + Default
            + AddAssign
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + PartialOrd
            + Copy
            + FromNumber,
    > TopKSoftmaxTrait<T> for TopKSoftmax<T>
{
    default fn compute(
        &self,
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        temperature: T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        thread_num: usize,
        input_topk_size: usize,
        top_k: usize,
    ) {
        kernel::scalar::truncated_topk_softmax::truncated_topk_softmax(
            input_values_ptr,
            input_indices_ptr,
            temperature,
            output_values_ptr,
            output_indices_ptr,
            thread_num,
            input_topk_size,
            top_k,
        );
    }
}

impl TopKSoftmaxTrait<f16> for TopKSoftmax<f16> {
    fn compute(
        &self,
        input_indices_ptr: *const usize,
        input_values_ptr: *const f16,
        temperature: f16,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut f16,
        thread_num: usize,
        input_topk_size: usize,
        top_k: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        {
            kernel::x86_64::f16_512::truncated_topk_softmax::truncated_topk_softmax(
                input_values_ptr,
                input_indices_ptr,
                temperature,
                output_values_ptr,
                output_indices_ptr,
                thread_num,
                input_topk_size,
                top_k,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        {
            let temp = temperature as f32;
            let total_candidates = thread_num * input_topk_size;
            let mut heap = crate::kernel::common::heap::FixedMinHeap::new(
                output_values_ptr,
                output_indices_ptr,
                top_k,
            );

            unsafe {
                for i in 0..total_candidates {
                    let value = *input_values_ptr.add(i);
                    if !(value as f32).is_finite() {
                        continue;
                    }
                    let index = *input_indices_ptr.add(i);
                    heap.push(value, index);
                }
            }

            let len = heap.len();
            if len == 0 {
                return;
            }
            heap.sort_desc();

            unsafe {
                let max_val = (*output_values_ptr.add(0)) as f32;
                let mut total_sum = 0.0f32;

                for i in 0..len {
                    let val = ((*output_values_ptr.add(i)) as f32 - max_val) / temp;
                    let exp_val = val.exp();
                    *output_values_ptr.add(i) = exp_val as f16;
                    total_sum += exp_val;
                }

                for i in 0..len {
                    let val = *output_values_ptr.add(i) as f32;
                    *output_values_ptr.add(i) = (val / total_sum) as f16;
                }
            }
        }
    }
}

impl TopKSoftmaxTrait<f32> for TopKSoftmax<f32> {
    fn compute(
        &self,
        input_indices_ptr: *const usize,
        input_values_ptr: *const f32,
        temperature: f32,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut f32,
        thread_num: usize,
        input_topk_size: usize,
        top_k: usize,
    ) {
        kernel::x86_64::f32_256::truncated_topk_softmax::truncated_topk_softmax(
            input_values_ptr,
            input_indices_ptr,
            temperature,
            output_values_ptr,
            output_indices_ptr,
            thread_num,
            input_topk_size,
            top_k,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::runtime::scheduling::SequenceSlice;
    use crate::runtime::{Phase, SequenceState};
    use approx::assert_ulps_eq;

    #[test]
    fn test_topk_softmax_f32() {
        let sequence_length = 2;
        let batch_size = 2;
        let top_k = 8;
        let thread_num = 4;
        let eos_id = 100;

        let total_candidates_per_item = top_k * thread_num;
        let input_len = batch_size * total_candidates_per_item;

        let mut input_values = Vec::<f32>::with_capacity(input_len);
        let mut input_indices = Vec::<usize>::with_capacity(input_len);
        let mut user_records_vec = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            user_records_vec.push(SequenceState {
                filling_length: 0,
                sequence_index: 1,
                kv_index: 1,
                phase: Phase::Decode,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            });
            for j in 0..total_candidates_per_item {
                input_values.push(5.0 - (j as f32 * 0.1) - (i as f32));
                input_indices.push(i * 1000 + j);
            }
        }

        let mut batch_list = user_records_vec;

        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut decode_lists = Vec::with_capacity(thread_num);
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
                    last_token_flag: true,
                });
            }
            decode_lists.push(slices);
        }
        let decode_list = decode_lists.iter().flatten().cloned().collect::<Vec<_>>();

        let mut output_values = vec![0.0f32; batch_size * top_k];
        let mut output_indices = vec![0; batch_size * top_k];
        let mut output_sequences = vec![0; batch_size * sequence_length];
        let mut batch_temperature = vec![1.0f32; batch_size];

        let operator = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            top_k,
            top_k,
            vec![eos_id],
        );

        for i in 0..thread_num {
            operator.run(
                batch_size,
                batch_size,
                thread_num,
                i,
                &[],
                &decode_list,
                &mut batch_list,
            );
        }

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

            let topk = &paired[..top_k];
            let max_val = topk[0].0;
            let denom: f32 = topk.iter().map(|(v, _)| (v - max_val).exp()).sum();

            let expected_probs: Vec<f32> = topk
                .iter()
                .map(|(v, _)| (v - max_val).exp() / denom)
                .collect();
            let expected_indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();

            let output_vals_slice = &output_values[i * top_k..(i + 1) * top_k];
            let output_idx_slice = &output_indices[i * top_k..(i + 1) * top_k];

            assert_ulps_eq!(output_vals_slice, expected_probs.as_slice(), max_ulps = 4);
            assert_eq!(output_idx_slice, expected_indices.as_slice());
            assert_eq!(output_sequences[batch_size + i], expected_indices[0]);
            assert_eq!(batch_list[i].sequence_index, 1);
            assert_eq!(batch_list[i].kv_index, 2);
        }
    }

    #[test]
    fn test_topk_softmax_default_temperature() {
        let sequence_length = 2;
        let batch_size = 1;
        let top_k = 8;
        let thread_num = 1;
        let eos_id = 100;

        let input_indices = (10usize..18).collect::<Vec<_>>();
        let input_values = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut batch_list = vec![SequenceState {
            filling_length: 0,
            sequence_index: 1,
            kv_index: 1,
            phase: Phase::Decode,
            notify: std::sync::Arc::new(tokio::sync::Notify::new()),
        }];

        let decode_list = [SequenceSlice {
            batch_index: 0,
            sequence_index: 1,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
        }];

        let mut output_values = vec![0.0f32; batch_size * top_k];
        let mut output_indices = vec![0usize; batch_size * top_k];
        let mut output_sequences = vec![usize::MAX; batch_size * sequence_length];
        let mut batch_temperature = vec![1.0f32; batch_size];

        let operator = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            top_k,
            top_k,
            vec![eos_id],
        );

        operator.run(1, 1, thread_num, 0, &[], &decode_list, &mut batch_list);

        let max_val = 8.0f32;
        let expected: Vec<f32> = input_values.iter().map(|&v| (v - max_val).exp()).collect();
        let denom: f32 = expected.iter().sum();
        let expected: Vec<f32> = expected.into_iter().map(|v| v / denom).collect();

        assert_ulps_eq!(output_values.as_slice(), expected.as_slice(), max_ulps = 4);
        assert_eq!(output_indices, input_indices);
        assert_eq!(output_sequences[1], 10);
    }

    #[test]
    fn test_topk_softmax_skips_prefill_dummy_decode_list() {
        let sequence_length = 4;
        let batch_size = 1;
        let top_k = 2;
        let thread_num = 2;
        let eos_id = 100;

        let input_indices = vec![10usize, 11, 12, 13];
        let input_values = vec![1.0f32, 0.5, 0.25, 0.125];
        let mut batch_list = vec![SequenceState {
            filling_length: 0,
            sequence_index: 3,
            kv_index: 3,
            phase: Phase::Prefill,
            notify: std::sync::Arc::new(tokio::sync::Notify::new()),
        }];

        let decode_list = [SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 3,
            last_token_flag: false,
        }];

        let mut output_values = vec![f32::NAN; batch_size * top_k];
        let mut output_indices = vec![usize::MAX; batch_size * top_k];
        let mut output_sequences = vec![usize::MAX; batch_size * sequence_length];
        let mut batch_temperature = vec![1.0f32; batch_size];

        let operator = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            top_k,
            top_k,
            vec![eos_id],
        );

        operator.run(3, 1, thread_num, 0, &[], &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[0].sequence_index, 6);
        assert_eq!(batch_list[0].filling_length, 0);
        assert_eq!(batch_list[0].kv_index, 6);
        assert_eq!(output_indices, vec![usize::MAX; batch_size * top_k]);
        assert!(output_values.iter().all(|value| value.is_nan()));
        assert_eq!(
            output_sequences,
            vec![usize::MAX; batch_size * sequence_length]
        );
    }

    #[test]
    fn test_topk_softmax_skips_non_last_token_slice() {
        let sequence_length = 4;
        let batch_size = 1;
        let top_k = 2;
        let thread_num = 1;
        let eos_id = 100;

        let input_indices = vec![10usize, 11];
        let input_values = vec![1.0f32, 0.5];
        let mut batch_list = vec![SequenceState {
            filling_length: 0,
            sequence_index: 3,
            kv_index: 7,
            phase: Phase::Decode,
            notify: std::sync::Arc::new(tokio::sync::Notify::new()),
        }];

        let decode_list = [SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 1,
            last_token_flag: false,
        }];

        let mut output_values = vec![f32::NAN; batch_size * top_k];
        let mut output_indices = vec![usize::MAX; batch_size * top_k];
        let mut output_sequences = vec![usize::MAX; batch_size * sequence_length];
        let mut batch_temperature = vec![1.0f32; batch_size];

        let operator = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            top_k,
            top_k,
            vec![eos_id],
        );

        operator.run(0, 1, thread_num, 0, &[], &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[0].sequence_index, 3);
        assert_eq!(batch_list[0].kv_index, 7);
        assert_eq!(output_indices, vec![usize::MAX; batch_size * top_k]);
        assert!(output_values.iter().all(|value| value.is_nan()));
        assert_eq!(
            output_sequences,
            vec![usize::MAX; batch_size * sequence_length]
        );
    }

    #[test]
    fn test_topk_softmax_processes_completed_prefill_entry() {
        let sequence_length = 4;
        let batch_size = 1;
        let top_k = 8;
        let thread_num = 1;
        let eos_id = 100;

        let total_candidates_per_item = top_k * thread_num;
        let total_candidate_count = sequence_length * total_candidates_per_item;
        let mut input_indices = vec![0usize; total_candidate_count];
        let mut input_values = vec![0.0f32; total_candidate_count];
        for index in 0..total_candidates_per_item {
            input_indices[index] = 10usize + index;
            input_values[index] = 5.0f32 - index as f32 * 0.1;
        }
        let mut batch_list = vec![SequenceState {
            filling_length: 3,
            sequence_index: 0,
            kv_index: 0,
            phase: Phase::Prefill,
            notify: std::sync::Arc::new(tokio::sync::Notify::new()),
        }];

        let decode_list = [SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 3,
            last_token_flag: true,
        }];

        let mut output_values = vec![0.0f32; sequence_length * top_k];
        let mut output_indices = vec![0usize; sequence_length * top_k];
        let mut output_sequences = vec![usize::MAX; batch_size * sequence_length];
        let mut batch_temperature = vec![1.0f32; batch_size];

        let operator = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            top_k,
            top_k,
            vec![eos_id],
        );

        operator.run(3, 1, thread_num, 0, &[], &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[0].sequence_index, 3);
        assert_eq!(batch_list[0].filling_length, 0);
        assert_eq!(batch_list[0].kv_index, 4);
        assert_eq!(output_indices[0], 10);
        assert_eq!(output_sequences[3], 10);
    }

    #[test]
    fn test_topk_softmax_advances_partial_prefill_without_output() {
        let sequence_length = 4;
        let batch_size = 1;
        let top_k = 2;
        let thread_num = 1;
        let eos_id = 100;

        let input_indices = vec![10usize, 11, 12, 13];
        let input_values = vec![1.0f32, 0.5, 0.25, 0.125];
        let mut batch_list = vec![SequenceState {
            filling_length: 4,
            sequence_index: 2,
            kv_index: 2,
            phase: Phase::Prefill,
            notify: std::sync::Arc::new(tokio::sync::Notify::new()),
        }];

        let decode_list = [SequenceSlice {
            batch_index: 0,
            sequence_index: 2,
            token_start_index: 0,
            length: 2,
            last_token_flag: false,
        }];

        let mut output_values = vec![f32::NAN; batch_size * top_k];
        let mut output_indices = vec![usize::MAX; batch_size * top_k];
        let mut output_sequences = vec![usize::MAX; batch_size * sequence_length];
        let mut batch_temperature = vec![1.0f32; batch_size];

        let operator = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            top_k,
            top_k,
            vec![eos_id],
        );

        operator.run(2, 0, thread_num, 0, &[], &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Prefill);
        assert_eq!(batch_list[0].sequence_index, 4);
        assert_eq!(batch_list[0].filling_length, 2);
        assert_eq!(batch_list[0].kv_index, 4);
        assert_eq!(output_indices, vec![usize::MAX; batch_size * top_k]);
        assert!(output_values.iter().all(|value| value.is_nan()));
        assert_eq!(
            output_sequences,
            vec![usize::MAX; batch_size * sequence_length]
        );
    }

    #[test]
    fn test_topk_softmax_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        let sequence_length = 2;
        let batch_size = 2;
        let top_k = 8;
        let thread_num = 4;
        let eos_id = 100;

        let total_candidates_per_item = top_k * thread_num;
        let input_len = batch_size * total_candidates_per_item;

        let mut input_values = Vec::<f16>::with_capacity(input_len);
        let mut input_indices = Vec::<usize>::with_capacity(input_len);
        let mut user_records_vec = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            user_records_vec.push(SequenceState {
                filling_length: 0,
                sequence_index: 1,
                kv_index: 1,
                phase: Phase::Decode,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            });
            for j in 0..total_candidates_per_item {
                let val = 5.0 - (j as f32 * 0.1) - (i as f32);
                input_values.push(val as f16);
                input_indices.push(i * 1000 + j);
            }
        }

        let mut batch_list = user_records_vec;

        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut decode_lists = Vec::with_capacity(thread_num);
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
                    last_token_flag: true,
                });
            }
            decode_lists.push(slices);
        }
        let decode_list = decode_lists.iter().flatten().cloned().collect::<Vec<_>>();

        let mut output_values = vec![0.0 as f16; batch_size * top_k];
        let mut output_indices = vec![0; batch_size * top_k];
        let mut output_sequences = vec![0; batch_size * sequence_length];
        let mut batch_temperature = vec![1.0f16; batch_size];

        let operator = TopKSoftmax::<f16>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            top_k,
            top_k,
            vec![eos_id],
        );

        for i in 0..thread_num {
            operator.run(
                batch_size,
                batch_size,
                thread_num,
                i,
                &[],
                &decode_list,
                &mut batch_list,
            );
        }

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

            let topk = &paired[..top_k];
            let max_val = topk[0].0 as f32;

            let topk_f32: Vec<(f32, usize)> =
                topk.iter().map(|(v, idx)| (*v as f32, *idx)).collect();
            let denom: f32 = topk_f32.iter().map(|(v, _)| (v - max_val).exp()).sum();

            let expected_probs: Vec<f32> = topk_f32
                .iter()
                .map(|(v, _)| (v - max_val).exp() / denom)
                .collect();
            let expected_indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();

            let output_vals_slice = &output_values[i * top_k..(i + 1) * top_k];
            let output_idx_slice = &output_indices[i * top_k..(i + 1) * top_k];

            for k in 0..top_k {
                let out_val = output_vals_slice[k] as f32;
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
            assert_eq!(batch_list[i].sequence_index, 1);
            assert_eq!(batch_list[i].kv_index, 2);
        }
    }
}
