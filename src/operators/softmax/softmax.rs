use std::ops::{AddAssign, Div, Sub};
use std::ptr;

use crate::num_traits::Exp;
use crate::num_traits::FromNumber;
use crate::operators::assign::assign;
use crate::operators::traits::SoftmaxMapTrait;
use crate::runtime::session::Phase;
use crate::runtime::session::SlotState;
use crate::runtime::SequenceSlice;

/// Plain temperature softmax over a full row of logits.
///
/// Unlike `TopKSoftmax` this operator performs no top-k selection, no min-P /
/// top-P filtering and no sampling: it turns every candidate row into a proper
/// probability distribution. The three softmax steps (exp, sum, normalize) all
/// run in-place on the same buffer, so `input_values_ptr` is read as logits and
/// written back as probabilities.
#[derive(Clone)]
pub struct Softmax<T> {
    // In-place buffer: [lift_size, vocab_size] logits -> probabilities.
    input_values_ptr: *mut T,
    // [batch] per-request temperature.
    batch_temperature: *mut T,
    sequence_stride: usize,
    vocab_size: usize,
}

// Softmax carries raw pointers across thread boundaries; the scheduler
// guarantees disjoint per-thread slices, so Send + Sync are safe here.
// (Mirrors TopKSoftmax without the per-field ConstPtr/MutPtr wrappers.)
unsafe impl<T: Send> Send for Softmax<T> {}
unsafe impl<T: Sync> Sync for Softmax<T> {}

impl<
        T: Exp
            + Default
            + AddAssign
            + Div<Output = T>
            + Sub<Output = T>
            + PartialOrd
            + Copy
            + FromNumber,
    > Softmax<T>
{
    pub fn new(
        input_values_ptr: *mut T,
        batch_temperature: *mut T,
        sequence_stride: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            input_values_ptr,
            batch_temperature,
            sequence_stride,
            vocab_size,
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
        let input_values_ptr = self.input_values_ptr;
        let batch_temperature_ptr = self.batch_temperature;
        let sequence_stride = self.sequence_stride;
        let vocab_size = self.vocab_size;
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

            // Softmax only produces a distribution for the last token of a step.
            if !slice.last_token_flag {
                continue;
            }

            let raw_temp = unsafe { *batch_temperature_ptr.add(batch_index) };
            let temperature = if raw_temp <= zero { one } else { raw_temp };

            let lift = slice.lift_index;
            let row_ptr = unsafe { input_values_ptr.add(lift * vocab_size) };

            // ── Three-step softmax, every step operating on the same memory ──
            self.compute(row_ptr, temperature, vocab_size);

            let write_sequence_index = record.next_sequence_index;
            if write_sequence_index >= sequence_stride {
                record.phase = Phase::Eos;
                record.notify.notify_one();
                continue;
            }

            // Advance decode cursor so the scheduler keeps pumping this slot.
            record.next_sequence_index += 1;
            record.sequence_length += 1;
            if write_sequence_index % 10 == 0 {
                record.notify.notify_one();
            }
        }
    }
}

// Three-step in-place softmax: exp -> sum -> normalize. Generic scalar path;
// numeric bounds come from crate::num_traits (mirrors TopKSoftmax / RMSMap).
impl<
        T: Exp
            + Default
            + AddAssign
            + Div<Output = T>
            + Sub<Output = T>
            + PartialOrd
            + Copy
            + FromNumber,
    > SoftmaxMapTrait<T> for Softmax<T>
{
    fn compute(&self, values_ptr: *mut T, temperature: T, length: usize) {
        if length == 0 {
            return;
        }
        let zero = T::default();

        // ── Step 1: numerically-stable exp, in place ──────────────────────
        // Find the row max, then overwrite every logit with
        // exp((x - max) / temperature).
        let mut max_val = unsafe { *values_ptr };
        for i in 1..length {
            let v = unsafe { *values_ptr.add(i) };
            if v > max_val {
                max_val = v;
            }
        }
        for i in 0..length {
            let v = unsafe { *values_ptr.add(i) };
            let e = ((v - max_val) / temperature).exp();
            unsafe { ptr::write(values_ptr.add(i), e) };
        }

        // ── Step 2: accumulate the sum over the exp'd buffer ──────────────
        let mut total_sum = zero;
        for i in 0..length {
            total_sum += unsafe { *values_ptr.add(i) };
        }

        // ── Step 3: normalize in place: x_i <- x_i / total_sum ────────────
        // Guard against a zero denominator (all logits filtered to 0) by
        // pinning the whole mass to the first entry.
        if total_sum <= zero {
            unsafe {
                ptr::write(values_ptr, T::from_f32(1.0));
            }
            for i in 1..length {
                unsafe { ptr::write(values_ptr.add(i), zero) };
            }
            return;
        }
        for i in 0..length {
            let v = unsafe { *values_ptr.add(i) };
            unsafe { ptr::write(values_ptr.add(i), v / total_sum) };
        }
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
    fn test_softmax_inplace_f32() {
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
            next_sequence_index: 1,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
            lift_index: 0,
        }];
        let mut batch_temperature = vec![1.0f32; batch_size];

        let operator = Softmax::<f32>::new(
            values.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            sequence_length,
            vocab_size,
        );

        operator.run(1, 1, 0, thread_num, 0, &decode_list, &mut slot_list);

        let expected = ref_softmax(&logits_row0, 1.0);
        assert_ulps_eq!(&values[0..vocab_size], expected.as_slice(), max_ulps = 4);
        // Row 1 untouched: not a last-token slice this round.
        assert_eq!(&values[vocab_size..2 * vocab_size], &logits_row1[..]);
        // Decode cursor advanced.
        assert_eq!(slot_list[0].next_sequence_index, 2);
    }

    #[test]
    fn test_softmax_temperature_f32() {
        let vocab_size = 4;
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let temperature = 2.0f32;

        let mut values = logits.to_vec();
        let mut slot_list = vec![decode_state(1, 1)];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 1,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
            lift_index: 0,
        }];
        let mut batch_temperature = vec![temperature];

        let operator = Softmax::<f32>::new(
            values.as_mut_ptr(),
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
    }

    #[test]
    fn test_softmax_skips_non_last_token() {
        let vocab_size = 4;
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let mut values = logits.to_vec();
        let mut slot_list = vec![decode_state(7, 3)];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: 1,
            last_token_flag: false,
            lift_index: 0,
        }];
        let mut batch_temperature = vec![1.0f32];

        let operator = Softmax::<f32>::new(
            values.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            4,
            vocab_size,
        );
        operator.run(0, 1, 0, 1, 0, &decode_list, &mut slot_list);

        // No softmax applied to a non-last-token slice.
        assert_eq!(&values[..], &logits[..]);
        assert_eq!(slot_list[0].phase, Phase::Decode);
        assert_eq!(slot_list[0].next_sequence_index, 7);
    }
}
