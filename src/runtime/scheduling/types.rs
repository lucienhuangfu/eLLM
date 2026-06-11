use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;

use super::sequence_slice::{DecodeList, SequenceSlice};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    Start,
    Prefill,
    Decode,
    Timeout,
    Eos,
}

#[derive(Debug, Clone)]
pub struct ScheduleTask {
    pub prefill_size: usize,
    pub decode_size: usize,
    pub prefill_list: Arc<Vec<Vec<SequenceSlice>>>,
    pub decode_list: Arc<DecodeList>,
    pub timestamp: Instant,
    pub task_id: u64,
}

impl ScheduleTask {
    pub fn new(
        prefill_size: usize,
        decode_size: usize,
        prefill_list: Vec<Vec<SequenceSlice>>,
        decode_list: DecodeList,
        task_id: u64,
    ) -> Self {
        Self {
            prefill_size,
            decode_size,
            prefill_list: Arc::new(prefill_list),
            decode_list: Arc::new(decode_list),
            timestamp: Instant::now(),
            task_id,
        }
    }
}

#[derive(Clone)]
pub struct SequenceState {
    pub sequence_index: usize,
    pub kv_index: usize,
    pub filling_length: usize,
    pub phase: Phase,
    pub notify: Arc<Notify>,
}

impl SequenceState {
    pub fn new_start_state() -> Self {
        Self {
            sequence_index: usize::MAX,
            kv_index: usize::MAX,
            filling_length: 0,
            phase: Phase::Start,
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn new_prefill_state(sequence_index: usize, filling_length: usize) -> Self {
        Self {
            sequence_index,
            kv_index: sequence_index,
            filling_length,
            phase: Phase::Prefill,
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn new_decode_state(sequence_index: usize, kv_index: usize) -> Self {
        Self {
            sequence_index,
            kv_index,
            filling_length: 0,
            phase: Phase::Decode,
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn transition_to_decode(&mut self) {
        self.phase = Phase::Decode;
        self.notify.notify_one();
    }

    pub fn transition_to_eos(&mut self) {
        self.phase = Phase::Eos;
        self.notify.notify_one();
    }

    pub fn transition_to_timeout(&mut self) {
        self.phase = Phase::Timeout;
        self.notify.notify_one();
    }

    pub fn reset_to_start(&mut self) {
        self.sequence_index = usize::MAX;
        self.kv_index = usize::MAX;
        self.filling_length = 0;
        self.phase = Phase::Start;
    }

    pub fn is_active(&self) -> bool {
        matches!(self.phase, Phase::Prefill | Phase::Decode)
    }

    pub fn is_available(&self) -> bool {
        matches!(self.phase, Phase::Start | Phase::Eos)
    }

    pub fn advance_sequence(&mut self, steps: usize) {
        self.sequence_index += steps;
        if self.phase == Phase::Prefill {
            self.filling_length = self.filling_length.saturating_sub(steps);
            if self.filling_length == 0 {
                self.transition_to_decode();
            }
        }
    }
}

impl Default for SequenceState {
    fn default() -> Self {
        Self::new_start_state()
    }
}
