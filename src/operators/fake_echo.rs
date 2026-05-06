//! Fake echo operator for integration tests and the standalone fake server.
//!
//! This operator is intentionally simple and visible:
//! - only thread `0` performs work
//! - it only reacts to `Phase::Prefill`
//! - it advances the request to `Phase::Eos` and wakes the waiting slot
//!
//! The goal is to prove that the runtime executed an operator without needing
//! a full model forward pass or any operator-owned state.

use crate::common::sequence_slice::SequenceSlice;
use crate::runtime::{Phase, SequenceState};

/// A tiny testing operator that completes prefill requests immediately.
#[derive(Clone)]
pub struct FakeEcho;

impl FakeEcho {
    /// Runs the fake operator for a batch.
    ///
    /// Only thread `0` performs the mutation. For each slot in `Prefill`:
    /// - the slot is marked `Eos`
    /// - the waiting notifier is triggered
    ///
    /// This keeps the response shape simple while still making the output
    /// visibly different from the original request lifecycle.
    pub fn run(
        &self,
        _prefill_list: &[Vec<SequenceSlice>],
        _decode_list: &[SequenceSlice],
        batch_list: &mut Vec<SequenceState>,
        thread_id: usize,
    ) {
        // ServingRunner 的 thread_id 从 0 开始，0 号线程负责推进 fake 完成。
        if thread_id != 0 {
            return;
        }

        for record in batch_list.iter_mut() {
            if matches!(record.phase, Phase::Prefill) {
                record.sequence_index = 0;
                record.kv_index = record.filling_length;
                record.filling_length = 0;
                record.phase = Phase::Eos;
                record.notify.notify_one();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FakeEcho;
    use crate::common::sequence_slice::SequenceSlice;
    use crate::runtime::{Phase, SequenceState};
    use std::sync::Arc;

    #[test]
    fn fake_echo_completes_prefill_and_finishes_request() {
        let echo = FakeEcho;
        let mut batch_list = vec![SequenceState {
            sequence_index: 0,
            kv_index: 0,
            filling_length: 3,
            phase: Phase::Prefill,
            notify: Arc::new(tokio::sync::Notify::new()),
        }];
        let prefill_list = vec![vec![SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 3,
            last_token_flag: true,
        }]];
        let decode_list = vec![];

        echo.run(&prefill_list, &decode_list, &mut batch_list, 0);

        assert_eq!(batch_list[0].phase, Phase::Eos);
        assert_eq!(batch_list[0].sequence_index, 0);
        assert_eq!(batch_list[0].kv_index, 3);
        assert_eq!(batch_list[0].filling_length, 0);
    }
}
