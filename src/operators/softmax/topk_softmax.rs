use std::f16;
use std::ops::{AddAssign, Div, Mul, Sub};
use std::ptr;

use crate::kernel;
use crate::num_traits::Exp;
use crate::num_traits::FromNumber;
use crate::operators::assign::assign;
use crate::operators::traits::TopKSoftmaxTrait;
use crate::runtime::session::Phase;
use crate::runtime::session::SlotState;
use crate::runtime::SequenceSlice;
use rand::Rng;

#[derive(Clone)]
pub struct TopKSoftmax<T> {
    input_indices_ptr: *const usize,
    input_values_ptr: *const T,
    output_indices_ptr: *mut usize,
    output_values_ptr: *mut T,
    output_sequences: *mut usize,
    batch_temperature: *mut T,
    sequence_stride: usize,
    input_top_k: usize,
    input_thread_capacity: usize,
    top_k: usize,
    top_p: T,
    min_p: T,
    top_p_enabled: bool,
    min_p_enabled: bool,
    do_sample: bool,
    eos_ids: Vec<usize>,
}

// TopKSoftmax carries raw pointers across thread boundaries; the scheduler guarantees
// disjoint per-thread slices, so Send + Sync are safe here. (Mirrors the previous
// ConstPtr/MutPtr wrappers without the per-field .ptr indirection.)
unsafe impl<T: Send> Send for TopKSoftmax<T> {}
unsafe impl<T: Sync> Sync for TopKSoftmax<T> {}

impl<
        T: Exp
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
    #[inline]
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
        let zero = T::default();
        let one = T::from_f32(1.0);
        Self {
            input_indices_ptr,
            input_values_ptr,
            output_indices_ptr,
            output_values_ptr,
            output_sequences,
            batch_temperature,
            sequence_stride,
            input_top_k,
            input_thread_capacity: 0,
            top_k,
            top_p,
            min_p,
            top_p_enabled: top_p > zero && top_p < one,
            min_p_enabled: min_p > zero,
            do_sample,
            eos_ids,
        }
    }

    #[inline]
    pub fn with_input_thread_capacity(mut self, input_thread_capacity: usize) -> Self {
        self.input_thread_capacity = input_thread_capacity;
        self
    }

    pub fn run(
        &self,
        _prefill_size: usize,
        _lift_size: usize,
        _total_size: usize,
        thread_num: usize,
        thread_id: usize,
        computing_slices: &[SequenceSlice],
        slot_list: &mut Vec<SlotState>,
    ) {
        let Some((begin, end)) = assign(computing_slices.len(), thread_num, thread_id) else {
            return;
        };
        let slices = &computing_slices[begin..end];

        // Hoist all struct reads out of the loop (single load each)
        let input_indices_ptr = self.input_indices_ptr;
        let input_values_ptr = self.input_values_ptr;
        let output_indices_ptr = self.output_indices_ptr;
        let output_values_ptr = self.output_values_ptr;
        let output_sequences_ptr = self.output_sequences;
        let batch_temperature_ptr = self.batch_temperature;
        let input_thread_capacity = self.input_thread_capacity.max(thread_num);
        let input_top_k = self.input_top_k;
        let top_k = self.top_k;
        let sequence_stride = self.sequence_stride;
        let eos_ids = self.eos_ids.as_slice();
        let one = T::from_f32(1.0);
        let zero = T::default();

        for slice in slices {
            let batch_index = slice.batch_index;
            let slice_length = slice.length;
            // Safety: scheduler guarantees batch_index < slot_list.len()
            let record = unsafe { slot_list.get_unchecked_mut(batch_index) };

            // Prefill: advance sequence index for every chunk. Per scheduler
            // invariant, phase transition to Decode can only happen when the
            // last prefill chunk is consumed (last_token_flag=true).
            // Decode slices are always last_token_flag=true.
            if matches!(record.phase, Phase::Prefill) {
                record.next_sequence_index += slice_length;
                if slice.last_token_flag && record.filling_length() == 0 {
                    record.phase = Phase::Decode;
                }
            }
            
            if !slice.last_token_flag {
                continue;
            }

            // ── Softmax + sampling (hot path for last-token slices only) ──
            let raw_temp = unsafe { *batch_temperature_ptr.add(batch_index) };
            let temperature = if raw_temp <= zero { one } else { raw_temp };

            let lift = slice.lift_index;
            let input_stride = lift * input_top_k * input_thread_capacity;
            let output_stride = lift * top_k;

            self.compute(
                unsafe { input_indices_ptr.add(input_stride) },
                unsafe { input_values_ptr.add(input_stride) },
                temperature,
                unsafe { output_indices_ptr.add(output_stride) },
                unsafe { output_values_ptr.add(output_stride) },
                thread_num,
                input_top_k,
                top_k,
            );

            let predict_token = unsafe {
                self.filter_and_sample(
                    output_indices_ptr.add(output_stride),
                    output_values_ptr.add(output_stride),
                )
            };

            let write_sequence_index = record.next_sequence_index;

            if write_sequence_index >= sequence_stride {
                record.phase = Phase::Eos;
                record.notify.notify_one();
                continue;
            }

            // Write sampled token + advance decode cursor + EOS handling
            let out_offset = batch_index * sequence_stride + write_sequence_index;
            unsafe {
                ptr::write(output_sequences_ptr.add(out_offset), predict_token);
            }

            record.next_sequence_index += 1;
            record.sequence_length += 1;

            let is_eos = eos_ids.contains(&predict_token);
            if is_eos {
                record.phase = Phase::Eos;
            }
            if is_eos || write_sequence_index % 10 == 0 {
                record.notify.notify_one();
            }
        }
    }

    /// Single-pass min-P + top-P + normalize + (optional) sample.
    ///
    /// Returns the sampled (or greedy argmax) token id. Normalized
    /// probabilities are written back in-place for downstream inspection.
    unsafe fn filter_and_sample(
        &self,
        output_indices_ptr: *mut usize,
        output_values_ptr: *mut T,
    ) -> usize {
        let len = self.top_k;
        let zero = T::default();
        let one = T::from_f32(1.0);

        // ── Pass 1: min-P filter (optional) + accumulate kept mass ────────
        let kept_mass = if self.min_p_enabled {
            let threshold = *output_values_ptr * self.min_p;
            let mut mass = zero;
            for i in 0..len {
                let p = *output_values_ptr.add(i);
                if p >= threshold {
                    mass += p;
                } else {
                    ptr::write(output_values_ptr.add(i), zero);
                }
            }
            mass
        } else {
            let mut mass = zero;
            for i in 0..len {
                mass += *output_values_ptr.add(i);
            }
            mass
        };

        // Fallback: everything was filtered out — pin all mass to #1 candidate
        if kept_mass <= zero {
            ptr::write(output_values_ptr, one);
            for i in 1..len {
                ptr::write(output_values_ptr.add(i), zero);
            }
            return *output_indices_ptr;
        }

        // ── Pass 2: top-P cutoff (optional) on post-min-P distribution ───
        let cutoff = if self.top_p_enabled {
            let target = kept_mass * self.top_p;
            let mut cum = zero;
            let mut cut = len;
            for i in 0..len {
                let p = *output_values_ptr.add(i);
                if p <= zero {
                    continue;
                }
                cum += p;
                if cum >= target {
                    cut = i + 1;
                    break;
                }
            }
            cut.max(1)
        } else {
            len
        };

        // ── Pass 3: sum selected, normalize, (optional) sample in one go ─
        //  We need the sum of [0..cutoff) first for normalization,
        //  then re-read for sampling. Keep two tight passes, they are short.
        let mut selected_mass = zero;
        for i in 0..cutoff {
            selected_mass += *output_values_ptr.add(i);
        }

        if selected_mass <= zero {
            ptr::write(output_values_ptr, one);
            for i in 1..len {
                ptr::write(output_values_ptr.add(i), zero);
            }
            return *output_indices_ptr;
        }

        let inv_mass = one / selected_mass;
        for i in 0..cutoff {
            let p = *output_values_ptr.add(i) * inv_mass;
            ptr::write(output_values_ptr.add(i), p);
        }
        for i in cutoff..len {
            ptr::write(output_values_ptr.add(i), zero);
        }

        if !self.do_sample {
            return *output_indices_ptr;
        }

        // ── Sampling: draw once, single cumulative scan over [0..cutoff) ──
        let sample: T = T::from_f32(rand::thread_rng().gen::<f32>());
        let mut cum = zero;
        for i in 0..cutoff {
            cum += *output_values_ptr.add(i);
            if sample <= cum {
                return *output_indices_ptr.add(i);
            }
        }
        *output_indices_ptr.add(cutoff - 1)
    }
}

// ── Trait impls: specialization via `default fn` = zero-cost runtime dispatch ──
// Generic path (all types): scalar kernel.
// Specialized paths: f16 (with AVX512FP16) and f32 use their respective SIMD kernels.
// This is exactly the user-requested pattern: `compute` in the trait impl contains
// only a single backend call, with no extra bookkeeping.

impl<
        T: Exp
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
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
    use crate::runtime::SequenceSlice;
    use crate::runtime::{Phase, SlotState};
    use approx::assert_ulps_eq;

    #[allow(dead_code)]
    const EMPTY_SLICES: &[SequenceSlice] = &[];

    fn decode_state(next_sequence_index: usize, prompt_length: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_decode(next_sequence_index, prompt_length);
        s
    }

    fn prefill_state(next_sequence_index: usize, filling_length: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_prefill(next_sequence_index, filling_length);
        s
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
        let mut slot_list = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            slot_list.push(decode_state(1, 1));
            for j in 0..total_candidates_per_item {
                input_values.push(5.0 - (j as f32 * 0.1) - (i as f32));
                input_indices.push(i * 1000 + j);
            }
        }

        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut decode_lists = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for batch_index in start..end {
                slices.push(SequenceSlice {
                    batch_index,
                    next_sequence_index: 1,
                    token_start_index: batch_index,
                    length: 1,
                    last_token_flag: true,
                    lift_index: batch_index,
                });
            }
            decode_lists.push(slices);
        }
        let decode_list: Vec<SequenceSlice> = decode_lists.iter().flatten().cloned().collect();

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
                0,
                thread_num,
                i,
                &decode_list,
                &mut slot_list,
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
            assert_eq!(slot_list[i].next_sequence_index, 2);
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
        let mut slot_list = vec![decode_state(1, 1)];

        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 1,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
            lift_index: 0,
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

        operator.run(1, 1, 0, thread_num, 0, &decode_list, &mut slot_list);

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
        // 符合真实调度约束：Prefill + !last_token 的分块只推进索引，
        // 不触发 Prefill→Decode（否则调度器会把它标成 last_token）。
        let sequence_length = 4;
        let batch_size = 1;
        let top_k = 2;
        let thread_num = 2;
        let eos_id = 100;

        let input_indices = vec![10usize, 11, 12, 13];
        let input_values = vec![1.0f32, 0.5, 0.25, 0.125];
        // prefill_state(start=3, filling=5) -> next=3, prompt=8, filling_length=5
        let mut slot_list = vec![prefill_state(3, 5)];

        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: 3,
            last_token_flag: false, // 非 last chunk，推进 3 后 filling_length=2 > 0
            lift_index: 0,
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

        operator.run(3, 1, 0, thread_num, 0, &decode_list, &mut slot_list);

        assert_eq!(slot_list[0].phase, Phase::Prefill);
        assert_eq!(slot_list[0].next_sequence_index, 6);
        assert_eq!(slot_list[0].filling_length(), 2);
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
        let mut slot_list = vec![decode_state(7, 3)];

        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: 1,
            last_token_flag: false,
            lift_index: 0,
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

        operator.run(0, 1, 0, thread_num, 0, &decode_list, &mut slot_list);

        assert_eq!(slot_list[0].phase, Phase::Decode);
        assert_eq!(slot_list[0].next_sequence_index, 7);
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
        let mut slot_list = vec![prefill_state(0, 3)];

        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: 3,
            last_token_flag: true,
            lift_index: 0,
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

        operator.run(3, 1, 0, thread_num, 0, &decode_list, &mut slot_list);

        assert_eq!(slot_list[0].phase, Phase::Decode);
        assert_eq!(slot_list[0].next_sequence_index, 4);
        assert_eq!(slot_list[0].filling_length(), 0);
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
        let mut slot_list = vec![prefill_state(2, 4)];

        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 2,
            token_start_index: 0,
            length: 2,
            last_token_flag: false,
            lift_index: 0,
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

        operator.run(2, 0, 0, thread_num, 0, &decode_list, &mut slot_list);

        assert_eq!(slot_list[0].phase, Phase::Prefill);
        assert_eq!(slot_list[0].next_sequence_index, 4);
        assert_eq!(slot_list[0].filling_length(), 2);
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
        let mut slot_list = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            slot_list.push(decode_state(1, 1));
            for j in 0..total_candidates_per_item {
                let val = 5.0 - (j as f32 * 0.1) - (i as f32);
                input_values.push(val as f16);
                input_indices.push(i * 1000 + j);
            }
        }

        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut decode_lists = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for batch_index in start..end {
                slices.push(SequenceSlice {
                    batch_index,
                    next_sequence_index: 1,
                    token_start_index: batch_index,
                    length: 1,
                    last_token_flag: true,
                    lift_index: batch_index,
                });
            }
            decode_lists.push(slices);
        }
        let decode_list: Vec<SequenceSlice> = decode_lists.iter().flatten().cloned().collect();

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
                0,
                thread_num,
                i,
                &decode_list,
                &mut slot_list,
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
            assert_eq!(slot_list[i].next_sequence_index, 2);
        }
    }
}
