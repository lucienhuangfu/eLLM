#[derive(Clone, Default, Debug)]
pub struct SequenceSlice {
    pub token_start_index: usize,
    pub batch_index: usize,
    pub next_sequence_index: usize,
    pub length: usize,
    pub last_token_flag: bool,
    pub lift_index: usize,
}

#[derive(Debug, Clone)]
pub struct ScheduleTask {
    pub prefill_size: usize,
    pub decode_size: usize,
    pub total_size: usize,
    pub slices: Vec<SequenceSlice>,
}

impl ScheduleTask {
    pub fn new(_thread_num: usize, max_batch_size: usize) -> Self {
        Self {
            prefill_size: 0,
            decode_size: 0,
            total_size: 0,
            slices: Vec::with_capacity(max_batch_size),
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.prefill_size = 0;
        self.decode_size = 0;
        self.total_size = 0;
        self.slices.clear();
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prefill_size == 0 && self.decode_size == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_task_lifecycle() {
        let mut task = ScheduleTask::new(2, 8);
        assert!(task.is_empty());

        task.prefill_size = 10;
        task.decode_size = 5;
        task.slices.push(SequenceSlice::default());

        assert!(!task.is_empty());

        task.reset();
        assert!(task.is_empty());
        assert!(task.slices.is_empty());
    }
}
