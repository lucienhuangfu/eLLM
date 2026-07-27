//! Fake generator operator for integration tests and the standalone fake server.
//!
//! # Overview
//!
//! `FakeEcho` is a minimal, self-contained operator designed to validate the runtime
//! execution pipeline **without** requiring a real model or learned weights. It cycles
//! through a fixed set of tokens during generation, producing readable and predictable
//! output for testing purposes.
//!
//! # Behaviour
//!
//! - All threads participate in the work (no idle threads).
//! - Handles both prefill and decode phases:
//!   - **Prefill**: Processes the last chunk of each prefill sequence and transitions
//!     the slot from `Phase::Prefill` → `Phase::Decode`.
//!   - **Decode**: Generates one token per sequence per step, cycling through `tokens`.
//! - When the write position reaches position 99:
//!   - Writes `eos_id` instead of the next token.
//!   - Transitions the request to `Phase::Eos`.
//!   - Calls `notify_one()` to wake the waiting slot.
//!
//! # Design Goal
//!
//! Prove that the runtime correctly schedules and executes an operator end-to-end
//! without needing a full model forward pass or any operator-owned state.

use crate::operators::assign::assign;
use crate::runtime::SequenceSlice;
use crate::runtime::{Phase, SlotState};

/// A lightweight fake generator operator that cycles through a fixed token sequence.
///
/// This is **not** a real neural-network operator. It exists solely for:
/// - Integration tests that need to verify the runtime scheduling pipeline.
/// - The standalone fake server that demonstrates serving without loading actual model weights.
#[derive(Clone)]
pub struct FakeEcho {
    sequences_ptr: *mut usize,
    sequence_stride: usize,
    eos_id: usize,
    tokens: Vec<usize>,
    max_gen_tokens: usize,
}

unsafe impl Send for FakeEcho {}
unsafe impl Sync for FakeEcho {}

impl FakeEcho {
    pub fn new(
        sequences_ptr: *mut usize,
        sequence_stride: usize,
        eos_id: usize,
        tokens: Vec<usize>,
        max_gen_tokens: usize,
    ) -> Self {
        Self {
            sequences_ptr,
            sequence_stride,
            eos_id,
            tokens,
            max_gen_tokens,
        }
    }

    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        thread_num: usize,
        thread_id: usize,
        _prefill_list: &[Vec<SequenceSlice>],
        decode_list: &[SequenceSlice],
        batch_list: &mut Vec<SlotState>,
    ) {
        let Some((begin, end)) = assign(decode_list.len(), thread_num, thread_id) else {
            return;
        };

        for slice in decode_list.iter().take(end).skip(begin) {
            self.process_slice(slice, batch_list);
        }
    }

    fn process_slice(&self, slice: &SequenceSlice, batch_list: &mut Vec<SlotState>) {
        let batch_index = slice.batch_index;
        let record = match batch_list.get_mut(batch_index) {
            Some(r) => r,
            None => return,
        };

        let prompt_length = if matches!(record.phase, Phase::Prefill) {
            slice.next_sequence_index + slice.length
        } else {
            record.prompt_length
        };

        let write_pos = if matches!(record.phase, Phase::Prefill) {
            slice.next_sequence_index + slice.length
        } else {
            record.next_sequence_index
        };

        let gen_step = write_pos - prompt_length;

        if gen_step >= self.max_gen_tokens {
            self.write_token(batch_index, write_pos, self.eos_id);
            record.next_sequence_index = write_pos + 1;
            record.phase = Phase::Eos;
            record.notify.notify_one();
        } else {
            let token = self.tokens[gen_step % self.tokens.len()];
            self.write_token(batch_index, write_pos, token);
            record.next_sequence_index = write_pos + 1;
            if matches!(record.phase, Phase::Prefill) {
                record.phase = Phase::Decode;
            }
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
    use super::FakeEcho;
    use crate::runtime::SequenceSlice;
    use crate::runtime::{Phase, SlotState};

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

    const PRE_TOKEN_COUNT: usize = 10;
    const EOS_ID: usize = 100;
    const MAX_GEN_TOKENS: usize = 10;

    fn fill_sequence(
        sequences: &mut [usize],
        batch_index: usize,
        stride: usize,
        offset: usize,
        count: usize,
    ) {
        for i in 0..count {
            sequences[batch_index * stride + offset + i] = (offset + i) % 50 + 1;
        }
    }

    #[test]
    fn fake_echo_generates_single_token_per_run() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        fill_sequence(&mut sequences, 0, sequence_stride, 0, PRE_TOKEN_COUNT);

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID, vec![42], MAX_GEN_TOKENS);
        let mut batch_list = vec![prefill_state(0, 0)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: PRE_TOKEN_COUNT,
            last_token_flag: true,
        }];

        echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[0].next_sequence_index, PRE_TOKEN_COUNT + 1);
        assert_eq!(batch_list[0].filling_length(), 0);
        assert_eq!(sequences[PRE_TOKEN_COUNT], 42);
    }

    #[test]
    fn fake_echo_cycles_through_tokens() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        fill_sequence(&mut sequences, 0, sequence_stride, 0, PRE_TOKEN_COUNT);

        let tokens = vec![1000, 2000, 3000];
        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID, tokens, 6);
        let mut batch_list = vec![decode_state(PRE_TOKEN_COUNT, PRE_TOKEN_COUNT)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: PRE_TOKEN_COUNT,
            last_token_flag: true,
        }];

        for i in 0..6 {
            echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);
        }

        assert_eq!(sequences[PRE_TOKEN_COUNT + 0], 1000);
        assert_eq!(sequences[PRE_TOKEN_COUNT + 1], 2000);
        assert_eq!(sequences[PRE_TOKEN_COUNT + 2], 3000);
        assert_eq!(sequences[PRE_TOKEN_COUNT + 3], 1000);
        assert_eq!(sequences[PRE_TOKEN_COUNT + 4], 2000);
        assert_eq!(sequences[PRE_TOKEN_COUNT + 5], 3000);
    }

    #[test]
    fn fake_echo_runs_on_all_threads() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        fill_sequence(&mut sequences, 0, sequence_stride, 0, PRE_TOKEN_COUNT);

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID, vec![7], MAX_GEN_TOKENS);
        let mut batch_list = vec![prefill_state(PRE_TOKEN_COUNT, 0)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: PRE_TOKEN_COUNT,
            last_token_flag: true,
        }];

        echo.run(0, 1, 2, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[0].next_sequence_index, PRE_TOKEN_COUNT + 1);
        assert_eq!(batch_list[0].filling_length(), 0);
        assert_eq!(sequences[PRE_TOKEN_COUNT], 7);
    }

    #[test]
    fn fake_echo_thread_assignment() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 3 * sequence_stride];
        for batch in 0..3 {
            fill_sequence(&mut sequences, batch, sequence_stride, 0, PRE_TOKEN_COUNT);
        }

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID, vec![99], MAX_GEN_TOKENS);
        let mut batch_list = vec![
            prefill_state(PRE_TOKEN_COUNT, 0),
            prefill_state(PRE_TOKEN_COUNT, 0),
            prefill_state(PRE_TOKEN_COUNT, 0),
        ];
        let prefill_list = vec![];
        let decode_list = vec![
            SequenceSlice {
                batch_index: 0,
                next_sequence_index: 0,
                token_start_index: 0,
                length: PRE_TOKEN_COUNT,
                last_token_flag: true,
            },
            SequenceSlice {
                batch_index: 1,
                next_sequence_index: 0,
                token_start_index: 0,
                length: PRE_TOKEN_COUNT,
                last_token_flag: true,
            },
            SequenceSlice {
                batch_index: 2,
                next_sequence_index: 0,
                token_start_index: 0,
                length: PRE_TOKEN_COUNT,
                last_token_flag: true,
            },
        ];

        echo.run(0, 3, 2, 0, &prefill_list, &decode_list, &mut batch_list);
        echo.run(0, 3, 2, 1, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[1].phase, Phase::Decode);
        assert_eq!(batch_list[2].phase, Phase::Decode);
        assert_eq!(sequences[PRE_TOKEN_COUNT], 99);
        assert_eq!(sequences[sequence_stride + PRE_TOKEN_COUNT], 99);
        assert_eq!(sequences[sequence_stride * 2 + PRE_TOKEN_COUNT], 99);
    }

    #[test]
    fn fake_echo_stops_after_max_gen_tokens_with_eos() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        let prompt_len = PRE_TOKEN_COUNT;
        let max_gen = 5;
        fill_sequence(&mut sequences, 0, sequence_stride, 0, prompt_len);
        for i in 0..prompt_len {
            assert_ne!(
                sequences[i], EOS_ID,
                "pre-written token at {i} must not be eos_id"
            );
        }

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID, vec![5], max_gen);
        let mut batch_list = vec![decode_state(prompt_len, prompt_len)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: prompt_len,
            last_token_flag: true,
        }];

        for i in 0..max_gen {
            echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);
            assert_eq!(batch_list[0].phase, Phase::Decode, "step {} should be decode", i);
            assert_eq!(sequences[prompt_len + i], 5);
        }

        echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Eos);
        assert_eq!(batch_list[0].next_sequence_index, prompt_len + max_gen + 1);
        assert_eq!(sequences[prompt_len + max_gen], EOS_ID);
    }

    #[test]
    fn fake_echo_handles_decode_phase() {
        let sequence_stride = 256;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        fill_sequence(&mut sequences, 0, sequence_stride, 0, PRE_TOKEN_COUNT);

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, EOS_ID, vec![88], MAX_GEN_TOKENS);
        let mut batch_list = vec![decode_state(PRE_TOKEN_COUNT, PRE_TOKEN_COUNT)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: PRE_TOKEN_COUNT,
            last_token_flag: true,
        }];

        echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(sequences[PRE_TOKEN_COUNT], 88);
    }
}
