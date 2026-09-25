//! Fake generator operator for integration tests and the standalone fake server.
//!
//! # Overview
//!
//! `FakeEcho` is a minimal, self-contained operator designed to validate the runtime
//! execution pipeline **without** requiring a real model or learned weights. It plays
//! back a fixed sequence of tokens during generation, producing predictable output
//! for testing purposes.
//!
//! # Behaviour
//!
//! - All threads participate in the work (no idle threads).
//! - Handles both prefill and decode phases:
//!   - **Prefill**: Processes the last chunk of each prefill sequence and transitions
//!     the slot from `Phase::Prefill` → `Phase::Decode`.
//!   - **Decode**: Generates one token per sequence per step from `tokens`.
//! - When the token sequence is exhausted:
//!   - Writes `eos_id` instead of the next token.
//!   - Transitions the request to `Phase::Eos`.
//!   - Calls `notify_one()` to wake the waiting slot.

use crate::operators::assign::assign;
use crate::runtime::SequenceSlice;
use crate::runtime::{Phase, SlotState};

/// A lightweight fake generator operator that plays back a fixed token sequence.
///
/// This is **not** a real neural-network operator. It exists solely for:
/// - Integration tests that need to verify the runtime scheduling pipeline.
/// - The standalone fake server that demonstrates serving without loading actual model weights.
/// - End-to-end testing of the streaming parser (reasoning, tool calls, etc.).
#[derive(Clone)]
pub struct FakeEcho {
    sequences_ptr: *mut usize,
    sequence_stride: usize,
    eos_id: usize,
    tokens: Vec<usize>,
}

unsafe impl Send for FakeEcho {}
unsafe impl Sync for FakeEcho {}

impl FakeEcho {
    pub fn new(
        sequences_ptr: *mut usize,
        sequence_stride: usize,
        eos_id: usize,
        tokens: Vec<usize>,
    ) -> Self {
        Self {
            sequences_ptr,
            sequence_stride,
            eos_id,
            tokens,
        }
    }

    pub fn run(
        &self,
        _total_size: usize,
        thread_num: usize,
        thread_id: usize,
        computing_slices: &[SequenceSlice],
        slot_list: &mut Vec<SlotState>,
    ) {
        let Some((begin, end)) = assign(computing_slices.len(), thread_num, thread_id) else {
            return;
        };

        for slice in computing_slices.iter().take(end).skip(begin) {
            self.process_slice(slice, slot_list);
        }
    }

    fn process_slice(&self, slice: &SequenceSlice, slot_list: &mut Vec<SlotState>) {
        let batch_index = slice.batch_index;
        let record = match slot_list.get_mut(batch_index) {
            Some(r) => r,
            None => return,
        };

        let is_prefill = matches!(record.phase, Phase::Prefill);
        let prompt_length = if is_prefill {
            slice.next_sequence_index + slice.length
        } else {
            record.prompt_length
        };
        let write_pos = if is_prefill {
            prompt_length
        } else {
            record.next_sequence_index
        };

        let gen_step = write_pos - prompt_length;

        let (token, is_eos) = if gen_step >= self.tokens.len() {
            (self.eos_id, true)
        } else {
            (self.tokens[gen_step], false)
        };

        self.write_token(batch_index, write_pos, token);
        record.next_sequence_index = write_pos + 1;

        if is_prefill {
            record.prompt_length = prompt_length;
        }

        if is_eos {
            record.phase = Phase::Eos;
            record.notify.notify_one();
        } else if is_prefill {
            record.phase = Phase::Decode;
        }
    }

    fn write_token(&self, batch_index: usize, offset: usize, token: usize) {
        unsafe {
            *self
                .sequences_ptr
                .add(batch_index * self.sequence_stride + offset) = token;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY_SLICES: &[SequenceSlice] = &[];
    const STRIDE: usize = 256;
    const PRE_COUNT: usize = 5;
    const EOS: usize = 100;

    fn prefill_state(next_seq: usize, filling_len: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_prefill(next_seq, filling_len);
        s
    }

    fn decode_state(next_seq: usize, prompt_len: usize) -> SlotState {
        let mut s = SlotState::idle();
        s.start_decode(next_seq, prompt_len);
        s
    }

    fn fill_seq(seq: &mut [usize], batch: usize, offset: usize, count: usize) {
        for i in 0..count {
            seq[batch * STRIDE + offset + i] = (offset + i) % 50 + 1;
        }
    }

    fn make_slice(batch: usize, len: usize) -> SequenceSlice {
        SequenceSlice {
            batch_index: batch,
            next_sequence_index: 0,
            token_start_index: 0,
            length: len,
            last_token_flag: true,
            lift_index: 0,
        }
    }

    #[test]
    fn generates_tokens_in_order() {
        let mut seq = vec![0usize; 2 * STRIDE];
        fill_seq(&mut seq, 0, 0, PRE_COUNT);

        let echo = FakeEcho::new(seq.as_mut_ptr(), STRIDE, EOS, vec![10, 20, 30, 40, 50]);
        let mut batch = vec![prefill_state(0, 0)];
        let dl = vec![make_slice(0, PRE_COUNT)];

        echo.run(dl.iter().map(|s| s.length).sum(), 1, 0, &dl, &mut batch);
        assert_eq!(batch[0].phase, Phase::Decode);
        assert_eq!(batch[0].next_sequence_index, PRE_COUNT + 1);
        assert_eq!(seq[PRE_COUNT], 10);

        for i in 1..5 {
            echo.run(dl.iter().map(|s| s.length).sum(), 1, 0, &dl, &mut batch);
            assert_eq!(seq[PRE_COUNT + i], (i + 1) * 10);
        }

        echo.run(dl.iter().map(|s| s.length).sum(), 1, 0, &dl, &mut batch);
        assert_eq!(batch[0].phase, Phase::Eos);
        assert_eq!(batch[0].next_sequence_index, PRE_COUNT + 6);
        assert_eq!(seq[PRE_COUNT + 5], EOS);
    }

    #[test]
    fn handles_empty_tokens() {
        let mut seq = vec![0usize; 2 * STRIDE];
        fill_seq(&mut seq, 0, 0, PRE_COUNT);

        let echo = FakeEcho::new(seq.as_mut_ptr(), STRIDE, EOS, vec![]);
        let mut batch = vec![decode_state(PRE_COUNT, PRE_COUNT)];
        let dl = vec![make_slice(0, PRE_COUNT)];

        echo.run(dl.iter().map(|s| s.length).sum(), 1, 0, &dl, &mut batch);

        assert_eq!(batch[0].phase, Phase::Eos);
        assert_eq!(batch[0].next_sequence_index, PRE_COUNT + 1);
        assert_eq!(seq[PRE_COUNT], EOS);
    }

    #[test]
    fn single_token_script() {
        let mut seq = vec![0usize; 2 * STRIDE];
        fill_seq(&mut seq, 0, 0, PRE_COUNT);

        let echo = FakeEcho::new(seq.as_mut_ptr(), STRIDE, EOS, vec![42]);
        let mut batch = vec![decode_state(PRE_COUNT, PRE_COUNT)];
        let dl = vec![make_slice(0, PRE_COUNT)];

        echo.run(dl.iter().map(|s| s.length).sum(), 1, 0, &dl, &mut batch);
        assert_eq!(batch[0].phase, Phase::Decode);
        assert_eq!(seq[PRE_COUNT], 42);

        echo.run(dl.iter().map(|s| s.length).sum(), 1, 0, &dl, &mut batch);
        assert_eq!(batch[0].phase, Phase::Eos);
        assert_eq!(seq[PRE_COUNT + 1], EOS);
    }

    #[test]
    fn runs_on_all_threads() {
        let mut seq = vec![0usize; 2 * STRIDE];
        fill_seq(&mut seq, 0, 0, PRE_COUNT);

        let echo = FakeEcho::new(seq.as_mut_ptr(), STRIDE, EOS, vec![7, 8, 9]);
        let mut batch = vec![prefill_state(PRE_COUNT, 0)];
        let dl = vec![make_slice(0, PRE_COUNT)];

        echo.run(dl.iter().map(|s| s.length).sum(), 2, 0, &dl, &mut batch);

        assert_eq!(batch[0].phase, Phase::Decode);
        assert_eq!(batch[0].next_sequence_index, PRE_COUNT + 1);
        assert_eq!(seq[PRE_COUNT], 7);
    }

    #[test]
    fn thread_assignment() {
        let mut seq = vec![0usize; 3 * STRIDE];
        for b in 0..3 {
            fill_seq(&mut seq, b, 0, PRE_COUNT);
        }

        let echo = FakeEcho::new(seq.as_mut_ptr(), STRIDE, EOS, vec![99, 98, 97]);
        let mut batch = vec![
            prefill_state(PRE_COUNT, 0),
            prefill_state(PRE_COUNT, 0),
            prefill_state(PRE_COUNT, 0),
        ];
        let dl = vec![
            make_slice(0, PRE_COUNT),
            make_slice(1, PRE_COUNT),
            make_slice(2, PRE_COUNT),
        ];

        echo.run(dl.iter().map(|s| s.length).sum(), 2, 0, &dl, &mut batch);
        echo.run(dl.iter().map(|s| s.length).sum(), 2, 1, &dl, &mut batch);

        assert_eq!(batch[0].phase, Phase::Decode);
        assert_eq!(batch[1].phase, Phase::Decode);
        assert_eq!(batch[2].phase, Phase::Decode);
        assert_eq!(seq[PRE_COUNT], 99);
        assert_eq!(seq[STRIDE + PRE_COUNT], 99);
        assert_eq!(seq[STRIDE * 2 + PRE_COUNT], 99);
    }
}
