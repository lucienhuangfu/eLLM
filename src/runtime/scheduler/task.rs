use super::sequence::SequenceSlice;

#[derive(Debug, Clone)]
pub struct ScheduleTask {
    pub prefill_size: usize,
    pub decode_size: usize,
    pub total_token_num: usize,
    pub prefilling_chunked_slices: Vec<Vec<SequenceSlice>>,
    pub slices: Vec<SequenceSlice>,
}

impl ScheduleTask {
    pub fn new(thread_num: usize, max_batch_size: usize) -> Self {
        let mut prefilling_chunked_slices = Vec::with_capacity(thread_num);
        for _ in 0..thread_num {
            prefilling_chunked_slices.push(Vec::with_capacity(max_batch_size));
        }
        Self {
            prefill_size: 0,
            decode_size: 0,
            total_token_num: 0,
            prefilling_chunked_slices,
            slices: Vec::with_capacity(max_batch_size),
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.prefill_size = 0;
        self.decode_size = 0;
        self.total_token_num = 0;
        for list in self.prefilling_chunked_slices.iter_mut() {
            list.clear();
        }
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
        task.prefilling_chunked_slices[0].push(SequenceSlice::default());
        task.slices.push(SequenceSlice::default());

        assert!(!task.is_empty());

        task.reset();
        assert!(task.is_empty());
        assert!(task.prefilling_chunked_slices[0].is_empty());
        assert!(task.slices.is_empty());
    }
}
