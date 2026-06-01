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

pub struct SequenceState {
    pub sequence_index: usize,
    pub kv_index: usize,
    pub filling_length: usize,
    pub phase: Phase,
    pub notify: Arc<Notify>,
}
#[derive(Clone, Debug)]
pub struct ScheduleTask {
    pub prefill_size: usize,
    pub decode_size: usize,
    pub prefill_list: Vec<Vec<SequenceSlice>>,
    pub decode_list: DecodeList,
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
            prefill_list,
            decode_list,
            timestamp: Instant::now(),
            task_id,
        }
    }
}
