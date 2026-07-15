use crate::runtime::batch::SequenceSlice;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchMode {
    Decode,
    Prefill,
    Mixed,
}

#[derive(Debug, Clone)]
pub struct ScheduleTask {
    pub mode: BatchMode,
    pub prefill_size: usize,
    pub decode_size: usize,
    pub prefill_list: Vec<Vec<SequenceSlice>>,
    pub decode_list: Vec<SequenceSlice>,
}

impl ScheduleTask {
    pub fn new(thread_num: usize, max_batch_size: usize) -> Self {
        let mut prefill_list = Vec::with_capacity(thread_num);
        for _ in 0..thread_num {
            prefill_list.push(Vec::with_capacity(max_batch_size));
        }
        Self {
            mode: BatchMode::Decode,
            prefill_size: 0,
            decode_size: 0,
            prefill_list,
            decode_list: Vec::with_capacity(max_batch_size),
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.mode = BatchMode::Decode;
        self.prefill_size = 0;
        self.decode_size = 0;
        for list in self.prefill_list.iter_mut() {
            list.clear();
        }
        self.decode_list.clear();
    }

    #[inline]
    pub fn sequence_count(&self) -> usize {
        self.decode_size
            + (if self.mode == BatchMode::Prefill || self.mode == BatchMode::Mixed {
                1
            } else {
                0
            })
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
        assert_eq!(task.sequence_count(), 0);

        task.mode = BatchMode::Mixed;
        task.prefill_size = 10;
        task.decode_size = 5;
        task.prefill_list[0].push(SequenceSlice::default());
        task.decode_list.push(SequenceSlice::default());

        assert!(!task.is_empty());
        assert_eq!(task.sequence_count(), 6);

        task.reset();
        assert!(task.is_empty());
        assert!(task.prefill_list[0].is_empty());
        assert!(task.decode_list.is_empty());
    }

    #[test]
    fn test_schedule_task_sequence_count() {
        let mut task = ScheduleTask::new(1, 8);

        task.mode = BatchMode::Decode;
        task.decode_size = 5;
        assert_eq!(task.sequence_count(), 5);

        task.mode = BatchMode::Prefill;
        task.prefill_size = 10;
        task.decode_size = 0;
        assert_eq!(task.sequence_count(), 1);

        task.mode = BatchMode::Mixed;
        task.prefill_size = 10;
        task.decode_size = 3;
        assert_eq!(task.sequence_count(), 4);
    }
}
