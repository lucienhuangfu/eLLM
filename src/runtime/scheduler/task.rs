use std::sync::Arc;
use std::time::Instant;

use crate::runtime::state::sequence::{DecodeList, SequenceSlice};

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
