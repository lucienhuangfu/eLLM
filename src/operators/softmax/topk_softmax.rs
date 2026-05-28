use std::f16;
use std::ops::{AddAssign, Div, Mul, Sub};
use std::ptr;

use crate::num_traits::Exp;
use crate::num_traits::FromNumber;
use crate::num_traits::NegInfinity;
use crate::num_traits::Sqrt;
use crate::operators::send_sync_ptr::{ConstPtr, MutPtr};
use crate::runtime::SequenceSlice;
use crate::kernel;
use crate::operators::assign::assign;
use crate::operators::traits::TopKSoftmaxTrait;
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
    top_k: usize,
    top_k_simd: usize,
    thread_num: usize,
    simd_output_values: Box<[T]>,
    simd_output_indices: Box<[usize]>,
    top_p: T,
    min_p: T,
    do_sample: bool,
    eos_token_id_list: Vec<usize>,
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
            + FromNumber
            + NegInfinity,
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
        top_k: usize,
        top_k_simd: usize,
        thread_num: usize,
        top_p: T,
        min_p: T,
        do_sample: bool,
        eos_token_id_list: Vec<usize>,
    ) -> Self {
        let thread_num = thread_num.max(1);
        let simd_output_values = vec![T::default(); thread_num * top_k_simd].into_boxed_slice();
        let simd_output_indices = vec![0usize; thread_num * top_k_simd].into_boxed_slice();

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
            top_k,
            top_k_simd,
            thread_num,
            simd_output_values,
            simd_output_indices,
            top_p,
            min_p,
            do_sample,
            eos_token_id_list,
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

        assert!(thread_num <= self.thread_num);
        assert!(thread_id < thread_num);

        let Some((begin, end)) = assign(decode_list.len(), thread_num, thread_id) else {
            return;
        };

        unsafe {
            let input_indices_ptr = self.input_indices_ptr.ptr;
            let input_values_ptr = self.input_values_ptr.ptr;
            let output_indices_ptr = self.output_indices_ptr.ptr;
            let output_values_ptr = self.output_values_ptr.ptr;
            let output_sequences_ptr = self.output_sequences.ptr;
            let scratch_offset = thread_id * self.top_k_simd;
            let scratch_indices_ptr =
                self.simd_output_indices.as_ptr().add(scratch_offset) as *mut usize;
            let scratch_values_ptr = self.simd_output_values.as_ptr().add(scratch_offset) as *mut T;

            for (row_index, slice) in decode_list.iter().enumerate().take(end).skip(begin) {
                self.process_decode_slice(
                    row_index,
                    slice,
                    batch_list,
                    thread_num,
                    input_indices_ptr,
                    input_values_ptr,
                    output_indices_ptr,
                    output_values_ptr,
                    output_sequences_ptr,
                    scratch_indices_ptr,
                    scratch_values_ptr,
                );
            }
        }
    }

    unsafe fn process_decode_slice(
        &self,
        row_index: usize,
        slice: &SequenceSlice,
        batch_list: &mut Vec<SequenceState>,
        thread_num: usize,
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        output_sequences_ptr: *mut usize,
        scratch_indices_ptr: *mut usize,
        scratch_values_ptr: *mut T,
    ) {
        let batch_index = slice.batch_index;
        let slice_length = slice.length;
        if slice_length == 0 || batch_index >= batch_list.len() {
            return;
        }

        let record = &mut batch_list[batch_index];
        if matches!(record.phase, Phase::Prefill) {
            record.sequence_index = record.sequence_index.saturating_add(slice_length);
            record.kv_index = record.kv_index.saturating_add(slice_length);
            record.filling_length = record.filling_length.saturating_sub(slice_length);

            if record.filling_length == 0 {
                record.phase = Phase::Decode;
            }
        }

        if !slice.last_token_flag || !matches!(record.phase, Phase::Decode) {
            return;
        }

        let write_sequence_index = record.kv_index;
        if write_sequence_index >= self.sequence_stride {
            record.phase = Phase::Eos;
            record.notify.notify_one();
            return;
        }

        let input_stride = row_index * self.top_k * thread_num;
        let batch_temperature = *self.batch_temperature.ptr.add(batch_index);

        self.compute(
            input_indices_ptr.add(input_stride),
            input_values_ptr.add(input_stride),
            batch_temperature,
            scratch_indices_ptr,
            scratch_values_ptr,
            thread_num,
            self.top_k,
            self.top_k_simd,
        );

        let predict_token = self.filter_and_sample(
            self.top_k,
            self.top_k_simd,
            scratch_indices_ptr,
            scratch_values_ptr,
        );
        let out_offset = batch_index * self.sequence_stride + write_sequence_index;
        ptr::write(output_sequences_ptr.add(out_offset), predict_token);
        ptr::copy_nonoverlapping(
            scratch_indices_ptr,
            output_indices_ptr.add(row_index * self.top_k),
            self.top_k,
        );
        ptr::copy_nonoverlapping(
            scratch_values_ptr,
            output_values_ptr.add(row_index * self.top_k),
            self.top_k,
        );

        record.sequence_index = write_sequence_index;
        record.kv_index = record.kv_index.saturating_add(1);
        if self.eos_token_id_list.contains(&predict_token) {
            record.phase = Phase::Eos;
            record.notify.notify_one();
        }
    }

    unsafe fn filter_and_sample(
        &self,
        valid_len: usize,
        padded_len: usize,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
    ) -> usize {
        if valid_len == 0 {
            return *self.eos_token_id_list.first().unwrap_or(&0);
        }

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
        let write_fallback_distribution = || {
            ptr::write(output_values_ptr, one);
            for i in 1..valid_len {
                ptr::write(output_values_ptr.add(i), zero);
            }
            for i in valid_len..padded_len {
                ptr::write(output_values_ptr.add(i), zero);
            }
        };

        let mut kept_mass = zero;
        for i in 0..valid_len {
            let prob = *output_values_ptr.add(i);
            if prob >= min_prob_threshold {
                kept_mass += prob;
            } else {
                ptr::write(output_values_ptr.add(i), zero);
            }
        }

        if kept_mass <= zero {
            write_fallback_distribution();
            return *output_indices_ptr;
        }

        let mut cutoff = valid_len;
        if top_p_enabled {
            let target_mass = kept_mass * self.top_p;
            let mut cumulative = zero;
            for i in 0..valid_len {
                let prob = *output_values_ptr.add(i);
                if prob <= zero {
                    continue;
                }
                cumulative += prob;
                if cumulative >= target_mass {
                    cutoff = i + 1;
                    break;
                }
            }

            if cutoff == 0 {
                cutoff = 1;
            }
        }

        let mut selected_mass = zero;
        for i in 0..cutoff {
            selected_mass += *output_values_ptr.add(i);
        }

        if selected_mass <= zero {
            write_fallback_distribution();
            return *output_indices_ptr;
        }

        let inv_mass = one / selected_mass;
        for i in 0..valid_len {
            let prob = if i < cutoff {
                *output_values_ptr.add(i) * inv_mass
            } else {
                zero
            };
            ptr::write(output_values_ptr.add(i), prob);
        }
        for i in valid_len..padded_len {
            ptr::write(output_values_ptr.add(i), zero);
        }

        if !self.do_sample {
            return *output_indices_ptr;
        }

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
            + FromNumber
            + NegInfinity,
    > TopKSoftmaxTrait<T> for TopKSoftmax<T>
{
    default fn compute(
        &self,
        input_indices_ptr: *const usize,
        input_values_ptr: *const T,
        temperature: T,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
        // output_token_ptr: *mut usize,
        thread_num: usize,
        top_k: usize,
        top_k_simd: usize,
    ) {
        kernel::scalar::truncated_topk_softmax::truncated_topk_softmax(
            input_values_ptr,
            input_indices_ptr,
            temperature,
            // sums_ptr,
            output_values_ptr,
            output_indices_ptr,
            // output_token_ptr,
            thread_num,
            top_k,
            top_k_simd,
        );
    }
}

impl TopKSoftmaxTrait<f16> for TopKSoftmax<f16> {
    fn compute(
        &self,
        _input_indices_ptr: *const usize,
        _input_values_ptr: *const f16,
        _temperature: f16,
        _output_indices_ptr: *mut usize,
        _output_values_ptr: *mut f16,
        // output_token_ptr: *mut usize,
        _thread_num: usize,
        _top_k: usize,
        _top_k_simd: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
        kernel::x86_64::f16_512::truncated_topk_softmax::truncated_topk_softmax(
            input_values_ptr,
            input_indices_ptr,
            temperature,
            // sums_ptr,
            output_values_ptr,
            output_indices_ptr,
            // output_token_ptr,
            thread_num,
            top_k,
            top_k_simd,
        );
        /*
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512fp16")))]
        kernel::scalar::softmax::softmax(
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
        temperature: f32,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut f32,
        // output_token_ptr: *mut usize,
        thread_num: usize,
        top_k: usize,
        top_k_simd: usize,
    ) {
        kernel::x86_64::f32_256::truncated_topk_softmax::truncated_topk_softmax(
            input_values_ptr,
            input_indices_ptr,
            temperature,
            output_values_ptr,
            output_indices_ptr,
            // output_token_ptr,
            thread_num,
            top_k,
            top_k_simd,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::runtime::SequenceSlice;
    use crate::runtime::{Phase, SequenceState};
    use approx::assert_ulps_eq;
    use std::mem::size_of;

    fn top_k_simd_for<T>(top_k: usize) -> usize {
        let simd_width = if size_of::<T>() == 2 { 32 } else { 8 };
        top_k.div_ceil(simd_width) * simd_width
    }

    #[test]
    fn test_topk_softmax_derives_simd_padding_from_top_k() {
        let input_indices = vec![0usize; 8];
        let input_values = vec![0.0f32; 8];
        let mut output_indices = vec![0usize; 8];
        let mut output_values = vec![0.0f32; 8];
        let mut output_sequences = vec![0usize; 1];
        let mut batch_temperature = vec![1.0f32; 1];

        let op = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            1,
            5,
            top_k_simd_for::<f32>(5),
            2,
            1.0f32,
            0.0f32,
            false,
            vec![1],
        );

        assert_eq!(op.top_k, 5);
        assert_eq!(op.top_k_simd, 8);
        assert_eq!(op.thread_num, 2);

        let input_indices_f16 = vec![0usize; 32];
        let input_values_f16 = vec![0.0f16; 32];
        let mut output_indices_f16 = vec![0usize; 32];
        let mut output_values_f16 = vec![0.0f16; 32];
        let mut output_sequences_f16 = vec![0usize; 1];
        let mut batch_temperature_f16 = vec![1.0f16; 1];

        let op_f16 = TopKSoftmax::<f16>::new(
            input_indices_f16.as_ptr(),
            input_values_f16.as_ptr(),
            output_indices_f16.as_mut_ptr(),
            output_values_f16.as_mut_ptr(),
            output_sequences_f16.as_mut_ptr(),
            batch_temperature_f16.as_mut_ptr(),
            1,
            17,
            top_k_simd_for::<f16>(17),
            4,
            1.0f16,
            0.0f16,
            false,
            vec![1],
        );

        assert_eq!(op_f16.top_k, 17);
        assert_eq!(op_f16.top_k_simd, 32);
    }

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
                // Create some decreasing values for each batch item
                input_values.push(5.0 - (j as f32 * 0.1) - (i as f32));
                input_indices.push(i * 1000 + j); // Unique indices
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

        let sums = vec![0.0f32; batch_size];
        let mut batch_temperature = vec![1.0f32; batch_size];
        let mut output_values = vec![0.0f32; batch_size * top_k];
        let mut output_indices = vec![0; batch_size * top_k];
        let mut output_sequences = vec![0; batch_size * sequence_length];

        let operator = TopKSoftmax::<f32>::new(
            input_indices.as_ptr(),
            input_values.as_ptr(),
            output_indices.as_mut_ptr(),
            output_values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            top_k,
            top_k_simd_for::<f32>(top_k),
            thread_num,
            1.0f32,
            0.0f32,
            false,
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

        // Verification
        for i in 0..batch_size {
            let i_usize = i;
            let total_candidates_per_item_usize = total_candidates_per_item;
            let top_k_usize = top_k;

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

            let topk = &paired[..top_k_usize];
            let max_val = topk[0].0;
            let denom: f32 = topk.iter().map(|(v, _)| (v - max_val).exp()).sum();

            let expected_probs: Vec<f32> = topk
                .iter()
                .map(|(v, _)| (v - max_val).exp() / denom)
                .collect();
            let expected_indices: Vec<usize> = topk.iter().map(|(_, idx)| *idx).collect();

            let output_vals_slice =
                &output_values[i_usize * top_k_usize..(i_usize + 1) * top_k_usize];
            let output_idx_slice =
                &output_indices[i_usize * top_k_usize..(i_usize + 1) * top_k_usize];

            assert_ulps_eq!(output_vals_slice, expected_probs.as_slice(), max_ulps = 4);
            assert_eq!(output_idx_slice, expected_indices.as_slice());
            assert_eq!(
                output_sequences[(batch_size) + i_usize],
                expected_indices[0]
            );
            assert_eq!(batch_list[i_usize].sequence_index, 1);
            assert_eq!(batch_list[i_usize].kv_index, 2);
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
            top_k_simd_for::<f32>(top_k),
            thread_num,
            1.0f32,
            0.0f32,
            false,
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
    fn test_topk_softmax_applies_top_p_and_min_p() {
        let sequence_length = 2;
        let batch_size = 1;
        let top_k = 8;
        let thread_num = 1;
        let eos_id = 100;

        let input_indices = vec![10usize, 11, 12, 13, 14, 15, 16, 17];
        let input_values = vec![4.0f32, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0];
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
            top_k_simd_for::<f32>(top_k),
            thread_num,
            0.8f32,
            0.1f32,
            false,
            vec![eos_id],
        );

        operator.run(1, 1, thread_num, 0, &[], &decode_list, &mut batch_list);

        let max_val = 4.0f32;
        let probs: Vec<f32> = input_values.iter().map(|&v| (v - max_val).exp()).collect();
        let denom: f32 = probs.iter().sum();
        let probs: Vec<f32> = probs.into_iter().map(|v| v / denom).collect();
        let kept_mass = probs[0] + probs[1];
        let expected0 = probs[0] / kept_mass;
        let expected1 = probs[1] / kept_mass;

        assert_ulps_eq!(output_values[0], expected0, max_ulps = 4);
        assert_ulps_eq!(output_values[1], expected1, max_ulps = 4);
        assert_eq!(output_values[2], 0.0);
        assert_eq!(output_values[3], 0.0);
        assert_eq!(output_indices, input_indices);
        assert_eq!(output_sequences[1], 10);
    }

    #[test]
    fn test_topk_softmax_sampling_path_uses_filtered_token() {
        let sequence_length = 2;
        let batch_size = 1;
        let top_k = 8;
        let thread_num = 1;
        let eos_id = 100;

        let input_indices = vec![20usize, 21, 22, 23, 24, 25, 26, 27];
        let input_values = vec![4.0f32, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0];
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
            top_k_simd_for::<f32>(top_k),
            thread_num,
            0.5f32,
            0.0f32,
            true,
            vec![eos_id],
        );

        operator.run(1, 1, thread_num, 0, &[], &decode_list, &mut batch_list);

        assert_eq!(output_sequences[1], 20);
        assert_eq!(output_indices, input_indices);
        assert!((output_values[0] - 1.0).abs() < 1e-5);
        assert_eq!(output_values[1], 0.0);
        assert_eq!(output_values[2], 0.0);
        assert_eq!(output_values[3], 0.0);
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
            top_k_simd_for::<f32>(top_k),
            thread_num,
            1.0f32,
            0.0f32,
            false,
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
            top_k_simd_for::<f32>(top_k),
            thread_num,
            1.0f32,
            0.0f32,
            false,
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
            top_k_simd_for::<f32>(top_k),
            thread_num,
            1.0f32,
            0.0f32,
            false,
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
            top_k_simd_for::<f32>(top_k),
            thread_num,
            1.0f32,
            0.0f32,
            false,
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
                // Create some decreasing values for each batch item
                let val = 5.0 - (j as f32 * 0.1) - (i as f32);
                input_values.push(val as f16);
                input_indices.push(i * 1000 + j); // Unique indices
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
            top_k_simd_for::<f16>(top_k),
            thread_num,
            1.0f16,
            0.0f16,
            false,
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

        // Verification
        for i in 0..batch_size {
            let i_usize = i;
            let total_candidates_per_item_usize = total_candidates_per_item;
            let top_k_usize = top_k;

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

            let topk = &paired[..top_k_usize];
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
                &output_values[i_usize * top_k_usize..(i_usize + 1) * top_k_usize];
            let output_idx_slice =
                &output_indices[i_usize * top_k_usize..(i_usize + 1) * top_k_usize];

            for k in 0..top_k_usize {
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
            assert_eq!(
                output_sequences[(batch_size) + i_usize],
                expected_indices[0]
            );
            assert_eq!(batch_list[i_usize].sequence_index, 1);
            assert_eq!(batch_list[i_usize].kv_index, 2);
        }
    }
}
