//! Fake echo operator for integration tests and the standalone fake server.
//!
//! This operator is intentionally simple and visible:
//! - all threads perform work
//! - it only reacts to `Phase::Prefill`
//! - it reads the last token from decode_list sequences
//! - it copies that token 10 times into the sequences
//! - it writes eos_id at the end
//! - it advances the request to `Phase::Eos` and wakes the waiting slot
//!
//! The goal is to prove that the runtime executed an operator without needing
//! a full model forward pass or any operator-owned state.

use crate::runtime::state::sequence::SequenceSlice;
use crate::runtime::{Phase, SlotState};

#[derive(Clone)]
pub struct FakeEcho {
    sequences_ptr: *mut usize,
    sequence_stride: usize,
    eos_id: usize,
}

impl FakeEcho {
    pub fn new(sequences_ptr: *mut usize, sequence_stride: usize, eos_id: usize) -> Self {
        Self {
            sequences_ptr,
            sequence_stride,
            eos_id,
        }
    }

    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        _cpu_num: usize,
        _thread_id: usize,
        _prefill_list: &[Vec<SequenceSlice>],
        decode_list: &[SequenceSlice],
        batch_list: &mut Vec<SlotState>,
    ) {
        for slice in decode_list {
            let batch_index = slice.batch_index;
            let record = match batch_list.get_mut(batch_index) {
                Some(r) => r,
                None => continue,
            };

            if !matches!(record.phase, Phase::Prefill) {
                continue;
            }

            let last_token_index = slice.sequence_index + slice.length - 1;
            let last_token = unsafe {
                *self
                    .sequences_ptr
                    .add(batch_index * self.sequence_stride + last_token_index)
            };

            let write_start = slice.sequence_index + slice.length;
            for i in 0..10 {
                let write_index = batch_index * self.sequence_stride + write_start + i;
                unsafe {
                    *self.sequences_ptr.add(write_index) = last_token;
                }
            }

            let eos_write_index = batch_index * self.sequence_stride + write_start + 10;
            unsafe {
                *self.sequences_ptr.add(eos_write_index) = self.eos_id;
            }

            record.sequence_index = write_start + 11;
            record.kv_index = record.kv_index.saturating_add(11);
            record.filling_length = 0;
            record.phase = Phase::Eos;
            record.notify.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FakeEcho;
    use crate::runtime::state::sequence::SequenceSlice;
    use crate::runtime::{Phase, SlotState};
    use std::sync::Arc;

    #[test]
    fn fake_echo_copies_last_token_10_times_and_writes_eos() {
        let sequence_stride = 256;
        let eos_id = 100;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        sequences[0] = 1;
        sequences[1] = 2;
        sequences[2] = 3;

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, eos_id);
        let mut batch_list = vec![SlotState::new_prefill_state(0, 3)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 3,
            last_token_flag: true,
        }];

        echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Eos);
        assert_eq!(batch_list[0].sequence_index, 14);
        assert_eq!(batch_list[0].kv_index, 11);
        assert_eq!(batch_list[0].filling_length, 0);

        for i in 0..10 {
            assert_eq!(
                sequences[3 + i],
                3,
                "Token at position {} should be 3",
                3 + i
            );
        }
        assert_eq!(
            sequences[13], eos_id,
            "Token at position 13 should be eos_id"
        );
    }

    #[test]
    fn fake_echo_runs_on_all_threads() {
        let sequence_stride = 256;
        let eos_id = 100;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        sequences[0] = 1;
        sequences[1] = 2;

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, eos_id);
        let mut batch_list = vec![SlotState::new_prefill_state(2, 11)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 2,
            last_token_flag: true,
        }];

        echo.run(0, 1, 2, 1, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Eos);
        assert_eq!(batch_list[0].sequence_index, 13);
        assert_eq!(batch_list[0].kv_index, 13);
        assert_eq!(batch_list[0].filling_length, 0);

        for i in 0..10 {
            assert_eq!(
                sequences[2 + i],
                2,
                "Token at position {} should be 2",
                2 + i
            );
        }
        assert_eq!(
            sequences[12], eos_id,
            "Token at position 12 should be eos_id"
        );
    }

    #[test]
    fn fake_echo_ignores_non_prefill_phases() {
        let sequence_stride = 256;
        let eos_id = 100;
        let mut sequences = vec![0usize; 2 * sequence_stride];
        sequences[0] = 1;

        let echo = FakeEcho::new(sequences.as_mut_ptr(), sequence_stride, eos_id);
        let mut batch_list = vec![SlotState::new_decode_state(1, 1)];
        let prefill_list = vec![];
        let decode_list = vec![SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
        }];

        echo.run(0, 1, 1, 0, &prefill_list, &decode_list, &mut batch_list);

        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(sequences[1], 0);
    }
}
