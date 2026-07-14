use crate::runtime::state::sequence::{DecodeList, SequenceSlice};

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
    pub decode_list: DecodeList,
}

impl ScheduleTask {
    pub fn new() -> Self {
        Self {
            mode: BatchMode::Decode,
            prefill_size: 0,
            decode_size: 0,
            prefill_list: Vec::new(),
            decode_list: DecodeList::with_capacity(0),
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
    pub fn resize_prefill_list(&mut self, thread_num: usize) {
        self.prefill_list.resize_with(thread_num, || Vec::new());
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
    fn test_schedule_task_new() {
        let task = ScheduleTask::new();

        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.prefill_size, 0);
        assert_eq!(task.decode_size, 0);
        assert!(task.is_empty());
    }

    #[test]
    fn test_schedule_task_reset() {
        let mut task = ScheduleTask::new();
        task.mode = BatchMode::Mixed;
        task.prefill_size = 10;
        task.decode_size = 5;
        task.prefill_list = vec![vec![SequenceSlice::default(); 5]];
        task.decode_list.push(SequenceSlice::default());

        task.reset();

        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.prefill_size, 0);
        assert_eq!(task.decode_size, 0);
        assert!(task.prefill_list[0].is_empty());
        assert!(task.decode_list.is_empty());
        assert!(task.is_empty());
    }

    #[test]
    fn test_schedule_task_resize_prefill_list() {
        let mut task = ScheduleTask::new();
        task.resize_prefill_list(4);

        assert_eq!(task.prefill_list.len(), 4);
    }

    #[test]
    fn test_schedule_task_sequence_count_decode() {
        let mut task = ScheduleTask::new();
        task.mode = BatchMode::Decode;
        task.decode_size = 5;

        assert_eq!(task.sequence_count(), 5);
    }

    #[test]
    fn test_schedule_task_sequence_count_prefill() {
        let mut task = ScheduleTask::new();
        task.mode = BatchMode::Prefill;
        task.prefill_size = 10;

        assert_eq!(task.sequence_count(), 1);
    }

    #[test]
    fn test_schedule_task_sequence_count_mixed() {
        let mut task = ScheduleTask::new();
        task.mode = BatchMode::Mixed;
        task.prefill_size = 10;
        task.decode_size = 3;

        assert_eq!(task.sequence_count(), 4);
    }

    #[test]
    fn test_schedule_task_is_empty() {
        let empty_task = ScheduleTask::new();
        assert!(empty_task.is_empty());

        let mut decode_task = ScheduleTask::new();
        decode_task.decode_size = 5;
        assert!(!decode_task.is_empty());

        let mut prefill_task = ScheduleTask::new();
        prefill_task.prefill_size = 10;
        assert!(!prefill_task.is_empty());

        let mut mixed_task = ScheduleTask::new();
        mixed_task.prefill_size = 10;
        mixed_task.decode_size = 5;
        assert!(!mixed_task.is_empty());
    }

    #[test]
    fn test_schedule_task_decode_list_access() {
        let mut task = ScheduleTask::new();
        task.decode_list.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 10,
            token_start_index: 0,
            length: 1,
            last_token_flag: true,
        });
        task.decode_list.push(SequenceSlice {
            batch_index: 1,
            sequence_index: 20,
            token_start_index: 1,
            length: 1,
            last_token_flag: true,
        });

        assert_eq!(task.decode_list[0].batch_index, 0);
        assert_eq!(task.decode_list[0].sequence_index, 10);
        assert_eq!(task.decode_list[1].batch_index, 1);
        assert_eq!(task.decode_list[1].sequence_index, 20);
    }

    #[test]
    fn test_schedule_task_prefill_list_access() {
        let mut task = ScheduleTask::new();
        task.resize_prefill_list(1);
        task.prefill_list[0].push(SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 5,
            last_token_flag: false,
        });
        task.prefill_list[0].push(SequenceSlice {
            batch_index: 0,
            sequence_index: 5,
            token_start_index: 5,
            length: 5,
            last_token_flag: false,
        });

        assert_eq!(task.prefill_list[0].len(), 2);
        assert_eq!(task.prefill_list[0][0].length, 5);
        assert_eq!(task.prefill_list[0][1].length, 5);
    }
}
