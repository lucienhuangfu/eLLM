use std::ops::{AddAssign, Div, Mul, Sub};
use std::ptr;

use crate::num_traits::Exp;
use crate::num_traits::FromNumber;
use crate::operators::assign::assign;
use crate::runtime::session::Phase;
use crate::runtime::session::SlotState;
use crate::runtime::SequenceSlice;
use rand::prelude::*;

/// Temperature sampling over a full row of logits (ports DeepSpec
/// `sample_tokens` from `deepspec/utils/sampling.py`).
///
/// Two branches, selected by the per-request temperature:
/// - `temperature < 1e-5`: greedy. Single scan for argmax, the row is
///   rewritten as a one-hot distribution and the argmax index is the token.
/// - otherwise: temperature softmax followed by multinomial sampling. The
///   three softmax steps (exp with max shift, sum, normalize) stay in-place
///   on the same buffer, fused into three row passes: max / exp+sum /
///   normalize+sample. The normalized probability row is kept for downstream
///   inspection, and the sampled token id is written to `output_sequences`.
#[derive(Clone)]
pub struct SampleTokens<T> {
    // In-place buffer: [lift_size, vocab_size] logits -> probabilities, indexed
    // by the per-slice `lift_index` (renamed after HeadLookup's `*_token_ptr`).
    dspark_token_ptr: *mut T,
    // [batch, sequence_stride] sampled token ids.
    output_sequences: *mut usize,
    // [batch] per-request temperature.
    batch_temperature: *mut T,
    sequence_stride: usize,
    vocab_size: usize,
    // Speculative-decoding proposal window: total proposal length and the
    // index of the current proposal slot (starts at 0).
    proposal_len: usize,
    proposal_index: usize,
}

// SampleTokens carries raw pointers across thread boundaries; the scheduler
// guarantees disjoint per-thread slices, so Send + Sync are safe here.
// (Mirrors TopKSoftmax without the per-field ConstPtr/MutPtr wrappers.)
unsafe impl<T: Send> Send for SampleTokens<T> {}
unsafe impl<T: Sync> Sync for SampleTokens<T> {}

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
    > SampleTokens<T>
{
    pub fn new(
        dspark_token_ptr: *mut T,
        output_sequences: *mut usize,
        batch_temperature: *mut T,
        sequence_stride: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            dspark_token_ptr,
            output_sequences,
            batch_temperature,
            sequence_stride,
            vocab_size,
            proposal_len: 0,
            proposal_index: 0,
        }
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
        let dspark_token_ptr = self.dspark_token_ptr;
        let output_sequences_ptr = self.output_sequences;
        let batch_temperature_ptr = self.batch_temperature;
        let sequence_stride = self.sequence_stride;
        let vocab_size = self.vocab_size;
        // DeepSpec: temperature < 1e-5 selects the greedy argmax branch.
        let greedy_threshold = T::from_f32(1e-5);

        // Sampling only produces a token for the last token of a step, so the
        // row index into `dspark_token_ptr` is not the slice position: some
        // slices don't decode at all. The scheduler already folds that into
        // `lift_index` (a compact counter incremented only for last-token
        // slices), so it is used directly as the decode row index.
        for slice in slices {
            if !slice.last_token_flag {
                continue;
            }

            let batch_index = slice.batch_index;
            // Safety: scheduler guarantees batch_index < slot_list.len()
            let record = unsafe { slot_list.get_unchecked_mut(batch_index) };

            let raw_temp = unsafe { *batch_temperature_ptr.add(batch_index) };
            let row_ptr = unsafe { dspark_token_ptr.add(slice.lift_index * vocab_size) };

            let predict_token = if raw_temp < greedy_threshold {
                self.greedy_row(row_ptr)
            } else {
                self.sample_row(row_ptr, raw_temp)
            };

            let write_proposal_index = record.next_proposal_index;

            // Write sampled token + advance the proposal cursor.
            let out_offset = batch_index * sequence_stride + write_proposal_index;
            unsafe {
                ptr::write(output_sequences_ptr.add(out_offset), predict_token);
            }

            record.next_proposal_index += 1;
            // record.sequence_length += 1;
        }
    }

    /// Greedy branch: first-max argmax over the raw logits (matches
    /// `torch.argmax`), then rewrite the row as a one-hot distribution.
    fn greedy_row(&self, row_ptr: *mut T) -> usize {
        let vocab_size = self.vocab_size;
        if vocab_size == 0 {
            return 0;
        }
        let one = T::from_f32(1.0);
        let zero = T::default();

        let mut best = 0usize;
        let mut max_val = unsafe { *row_ptr };
        for i in 1..vocab_size {
            let v = unsafe { *row_ptr.add(i) };
            if v > max_val {
                max_val = v;
                best = i;
            }
        }
        for i in 0..vocab_size {
            unsafe { ptr::write(row_ptr.add(i), if i == best { one } else { zero }) };
        }
        best
    }

    /// Temperature softmax + multinomial sample, three row passes total.
    ///
    /// Pass 1 finds the row max; pass 2 writes exp((x - max) / temperature)
    /// in place while accumulating the total; pass 3 normalizes in place and
    /// samples at once: with `target = u * total` for `u ~ U(0,1)`, the first
    /// index whose unnormalized cumulative sum reaches `target` is drawn.
    /// The scan never breaks early so every entry still gets normalized.
    /// Returns the sampled token id (index into the row).
    fn sample_row(&self, row_ptr: *mut T, temperature: T) -> usize {
        let vocab_size = self.vocab_size;
        if vocab_size == 0 {
            return 0;
        }
        let zero = T::default();
        let one = T::from_f32(1.0);

        // ── Pass 1: row max ───────────────────────────────────────────────
        let mut max_val = unsafe { *row_ptr };
        for i in 1..vocab_size {
            let v = unsafe { *row_ptr.add(i) };
            if v > max_val {
                max_val = v;
            }
        }

        // ── Pass 2: numerically-stable exp in place, accumulate total ─────
        let mut total = zero;
        for i in 0..vocab_size {
            let v = unsafe { *row_ptr.add(i) };
            let e = ((v - max_val) / temperature).exp();
            unsafe { ptr::write(row_ptr.add(i), e) };
            total += e;
        }

        // Guard against a zero denominator (all logits underflowed to 0) by
        // pinning the whole mass to the first entry.
        if total <= zero {
            unsafe { ptr::write(row_ptr, one) };
            for i in 1..vocab_size {
                unsafe { ptr::write(row_ptr.add(i), zero) };
            }
            return 0;
        }

        // ── Pass 3: normalize in place + cumulative sample in one go ──────
        let u = T::from_f32(rand::rng().random::<f32>());
        let target = u * total;
        let mut cum = zero;
        let mut token = vocab_size - 1;
        let mut hit = false;
        for i in 0..vocab_size {
            let w = unsafe { *row_ptr.add(i) };
            unsafe { ptr::write(row_ptr.add(i), w / total) };
            if !hit {
                cum += w;
                if target <= cum {
                    token = i;
                    hit = true;
                }
            }
        }
        token
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::runtime::{Phase, SlotState};
    use approx::assert_ulps_eq;

    fn decode_state(next_sequence_index: usize, prompt_length: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_decode(next_sequence_index, prompt_length);
        s.next_proposal_index = next_sequence_index;
        s
    }

    fn ref_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits
            .iter()
            .map(|v| (v - max) / temperature)
            .map(f32::exp)
            .collect();
        let sum: f32 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    #[test]
    fn test_sample_tokens_inplace_f32() {
        let sequence_length = 2;
        let batch_size = 1;
        let vocab_size = 4;
        let thread_num = 1;

        // Two rows indexed by lift_index; only row 0 is a last-token slice.
        let logits_row0 = [1.0f32, 2.0, 3.0, 4.0];
        let logits_row1 = [0.5f32, 0.5, 1.0, 2.0];
        let mut values = vec![
            logits_row0[0],
            logits_row0[1],
            logits_row0[2],
            logits_row0[3],
            logits_row1[0],
            logits_row1[1],
            logits_row1[2],
            logits_row1[3],
        ];

        let mut slot_list = vec![decode_state(1, 1)];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            local_next_index: 1,
            global_start_index: 0,
            local_length: 1,
            last_token_flag: true,
            lift_index: 0,
        }];
        let mut batch_temperature = vec![1.0f32; batch_size];
        let mut output_sequences = vec![usize::MAX; batch_size * sequence_length];

        let operator = SampleTokens::<f32>::new(
            values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            vocab_size,
        );

        operator.run(1, 1, 0, thread_num, 0, &decode_list, &mut slot_list);

        // Normalized probability row kept: fused scan must not truncate
        // the normalize write-back after the sample hit.
        let expected = ref_softmax(&logits_row0, 1.0);
        assert_ulps_eq!(&values[0..vocab_size], expected.as_slice(), max_ulps = 4);
        // Row 1 untouched: not a last-token slice this round.
        assert_eq!(&values[vocab_size..2 * vocab_size], &logits_row1[..]);
        // A valid token was sampled and written at the decode cursor
        // (batch 0 * stride 2 + write index 1).
        assert_eq!(output_sequences[0], usize::MAX);
        assert!(output_sequences[1] < vocab_size);
        // Proposal cursor advanced.
        assert_eq!(slot_list[0].next_proposal_index, 2);
    }

    #[test]
    fn test_sample_tokens_temperature_f32() {
        let vocab_size = 4;
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let temperature = 2.0f32;

        let mut values = logits.to_vec();
        let mut slot_list = vec![decode_state(1, 1)];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            local_next_index: 1,
            global_start_index: 0,
            local_length: 1,
            last_token_flag: true,
            lift_index: 0,
        }];
        let mut batch_temperature = vec![temperature];
        let mut output_sequences = vec![usize::MAX; 2];

        let operator = SampleTokens::<f32>::new(
            values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            2,
            vocab_size,
        );
        operator.run(1, 1, 0, 1, 0, &decode_list, &mut slot_list);

        let expected = ref_softmax(&logits, temperature);
        assert_ulps_eq!(&values[0..vocab_size], expected.as_slice(), max_ulps = 4);

        // Sanity: probabilities sum to 1.
        let total: f32 = values[0..vocab_size].iter().sum();
        assert_ulps_eq!(total, 1.0f32, epsilon = 1e-6);
        assert!(output_sequences[1] < vocab_size);
    }

    #[test]
    fn test_sample_tokens_peaked_hits_argmax() {
        // Strongly peaked distribution: the multinomial draw lands on the
        // argmax index with overwhelming probability (~1 - e^-50).
        let vocab_size = 4;
        let logits = [0.0f32, 0.0, 50.0, 0.0];

        let mut values = logits.to_vec();
        let mut slot_list = vec![decode_state(1, 1)];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            local_next_index: 1,
            global_start_index: 0,
            local_length: 1,
            last_token_flag: true,
            lift_index: 0,
        }];
        let mut batch_temperature = vec![1.0f32];
        let mut output_sequences = vec![usize::MAX; 2];

        let operator = SampleTokens::<f32>::new(
            values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            2,
            vocab_size,
        );
        operator.run(1, 1, 0, 1, 0, &decode_list, &mut slot_list);

        assert_eq!(output_sequences[1], 2);
    }

    #[test]
    fn test_sample_tokens_greedy_f32() {
        // temperature < 1e-5 selects the greedy argmax branch: the row is
        // rewritten as one-hot and the argmax index is emitted as token.
        let vocab_size = 4;
        let logits = [1.0f32, 4.0, 3.0, 2.0];

        let mut values = logits.to_vec();
        let mut slot_list = vec![decode_state(1, 1)];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            local_next_index: 1,
            global_start_index: 0,
            local_length: 1,
            last_token_flag: true,
            lift_index: 0,
        }];
        let mut batch_temperature = vec![0.0f32];
        let mut output_sequences = vec![usize::MAX; 2];

        let operator = SampleTokens::<f32>::new(
            values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            2,
            vocab_size,
        );
        operator.run(1, 1, 0, 1, 0, &decode_list, &mut slot_list);

        assert_eq!(&values[..], &[0.0f32, 1.0, 0.0, 0.0][..]);
        assert_eq!(output_sequences[1], 1);
        assert_eq!(slot_list[0].next_proposal_index, 2);
    }

    #[test]
    fn test_sample_tokens_write_offset() {
        // Token offset must be batch_index * sequence_stride + write_index.
        let batch_size = 2;
        let sequence_stride = 4;
        let vocab_size = 4;

        // Row 0 (lift 0) belongs to batch 0, row 1 (lift 1) to batch 1.
        let logits = [50.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0];
        let mut values = logits.to_vec();
        // Batch 0 writes at next_sequence_index = 1, batch 1 at 2.
        let mut slot_list = vec![decode_state(1, 1), decode_state(2, 1)];
        let decode_list = vec![
            SequenceSlice {
                batch_index: 0,
                local_next_index: 1,
                global_start_index: 0,
                local_length: 1,
                last_token_flag: true,
                lift_index: 0,
            },
            SequenceSlice {
                batch_index: 1,
                local_next_index: 2,
                global_start_index: 0,
                local_length: 1,
                last_token_flag: true,
                lift_index: 1,
            },
        ];
        let mut batch_temperature = vec![0.0f32; batch_size];
        let mut output_sequences = vec![usize::MAX; batch_size * sequence_stride];

        let operator = SampleTokens::<f32>::new(
            values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_stride,
            vocab_size,
        );
        operator.run(2, 2, 0, 1, 0, &decode_list, &mut slot_list);

        // batch 0: 0 * 4 + 1 = 1, batch 1: 1 * 4 + 2 = 6.
        assert_eq!(output_sequences[1], 0);
        assert_eq!(output_sequences[6], 2);
        assert_eq!(output_sequences[0], usize::MAX);
        assert_eq!(output_sequences[5], usize::MAX);
    }

    #[test]
    fn test_sample_tokens_skips_non_last_token() {
        let vocab_size = 4;
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let mut values = logits.to_vec();
        let mut slot_list = vec![decode_state(7, 3)];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            local_next_index: 0,
            global_start_index: 0,
            local_length: 1,
            last_token_flag: false,
            lift_index: 0,
        }];
        let mut batch_temperature = vec![1.0f32];
        let mut output_sequences = vec![usize::MAX; 4];

        let operator = SampleTokens::<f32>::new(
            values.as_mut_ptr(),
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            4,
            vocab_size,
        );
        operator.run(0, 1, 0, 1, 0, &decode_list, &mut slot_list);

        // No sampling applied to a non-last-token slice.
        assert_eq!(&values[..], &logits[..]);
        assert_eq!(output_sequences[..], [usize::MAX; 4][..]);
        assert_eq!(slot_list[0].phase, Phase::Decode);
        assert_eq!(slot_list[0].next_sequence_index, 7);
        assert_eq!(slot_list[0].next_proposal_index, 7);
    }
}
