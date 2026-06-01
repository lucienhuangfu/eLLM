use crate::runtime::scheduling::sequence_slice::SequenceSlice;

pub(super) struct SliceScheduler {
    task_count: usize,
    total_tokens: usize,
    scheduled_tokens: usize,
    current_task: usize,
    current_task_remaining: usize,
}

impl SliceScheduler {
    pub(super) fn new(task_count: usize) -> Self {
        Self {
            task_count,
            total_tokens: 0,
            scheduled_tokens: 0,
            current_task: 0,
            current_task_remaining: 0,
        }
    }

    pub(super) fn set_task_count(&mut self, task_count: usize) {
        self.task_count = task_count;
    }

    pub(super) fn init(&mut self, total_tokens: usize) {
        self.total_tokens = total_tokens;
        self.scheduled_tokens = 0;
        self.current_task = 0;
        self.current_task_remaining = if total_tokens > 0 {
            self.quota_for(0)
        } else {
            0
        };
    }

    pub(super) fn is_done(&self) -> bool {
        self.scheduled_tokens >= self.total_tokens
    }

    pub(super) fn remaining_tokens(&self) -> usize {
        self.total_tokens.saturating_sub(self.scheduled_tokens)
    }

    fn active_task_count(&self) -> usize {
        let base_quota = self.total_tokens / self.task_count;
        if base_quota == 0 {
            self.total_tokens.min(self.task_count)
        } else {
            self.task_count
        }
    }

    fn quota_for(&self, task_index: usize) -> usize {
        let base_quota = self.total_tokens / self.task_count;
        let extra_quota = self.total_tokens % self.task_count;
        if task_index < extra_quota {
            base_quota + 1
        } else {
            base_quota
        }
    }

    fn advance_to_next_task(&mut self) {
        while self.current_task < self.active_task_count() && self.current_task_remaining == 0 {
            self.current_task += 1;
            if self.current_task < self.active_task_count() {
                self.current_task_remaining = self.quota_for(self.current_task);
            }
        }
    }

    pub(super) fn current_task_index(&mut self) -> Option<usize> {
        if self.is_done() {
            return None;
        }

        self.advance_to_next_task();

        if self.current_task >= self.active_task_count() {
            None
        } else {
            Some(self.current_task)
        }
    }

    pub(super) fn take(&mut self, max_take: usize) -> usize {
        if self.is_done() || max_take == 0 {
            return 0;
        }

        let available = self.total_tokens - self.scheduled_tokens;
        let take = max_take.min(available).min(self.current_task_remaining);
        if take == 0 {
            return 0;
        }

        self.scheduled_tokens += take;
        self.current_task_remaining -= take;
        take
    }

    pub(super) fn schedule_for_sequence(
        &mut self,
        batch_index: usize,
        sequence_index: usize,
        mut remaining: usize,
        token_offset: usize,
        slice_list: &mut Vec<Vec<SequenceSlice>>,
        token_count: &mut usize,
    ) {
        if self.is_done() {
            return;
        }

        let mut sequence_cursor = sequence_index;

        while remaining > 0 && !self.is_done() {
            let Some(task_index) = self.current_task_index() else {
                break;
            };

            let token_start_index = token_offset + self.scheduled_tokens;
            let take = self.take(remaining);
            if take == 0 {
                break;
            }

            slice_list[task_index].push(SequenceSlice {
                batch_index,
                sequence_index: sequence_cursor,
                token_start_index,
                length: take,
                last_token_flag: false,
            });

            *token_count += take;
            remaining -= take;
            sequence_cursor += take;
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct PrefillCandidate {
    pub(super) batch_index: usize,
    pub(super) sequence_index: usize,
    pub(super) remaining: usize,
}

#[cfg(test)]
mod tests {
    use super::SliceScheduler;
    use crate::runtime::scheduling::sequence_slice::SequenceSlice;

    fn make_slices(task_num: usize) -> Vec<Vec<SequenceSlice>> {
        (0..task_num).map(|_| Vec::new()).collect()
    }

    #[test]
    fn init_balances_a_long_run_across_tasks() {
        let mut scheduler = SliceScheduler::new(6);
        scheduler.init(47);

        let mut per_task = [0usize; 6];
        let mut trace = Vec::new();

        while let Some(task_index) = scheduler.current_task_index() {
            let taken = scheduler.take(usize::MAX);
            if taken == 0 {
                break;
            }
            per_task[task_index] += taken;
            trace.push((task_index, taken));
        }

        assert_eq!(per_task, [8, 8, 8, 8, 8, 7]);
        assert_eq!(trace, vec![(0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 7)]);
        assert!(scheduler.is_done());
        assert_eq!(scheduler.scheduled_tokens, 47);
        assert_eq!(scheduler.remaining_tokens(), 0);
        assert_eq!(scheduler.current_task_index(), None);
    }

    #[test]
    fn init_with_more_tasks_than_tokens_activates_only_necessary_tasks() {
        let mut scheduler = SliceScheduler::new(8);
        scheduler.init(5);

        let mut trace = Vec::new();
        while let Some(task_index) = scheduler.current_task_index() {
            let taken = scheduler.take(1);
            if taken == 0 {
                break;
            }
            trace.push((task_index, taken));
        }

        assert_eq!(trace, vec![(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]);
        assert!(scheduler.is_done());
        assert_eq!(scheduler.scheduled_tokens, 5);
        assert_eq!(scheduler.remaining_tokens(), 0);
        assert_eq!(scheduler.current_task_index(), None);
    }

    #[test]
    fn set_task_count_changes_the_quota_shape_before_init() {
        let mut scheduler = SliceScheduler::new(2);
        scheduler.set_task_count(5);
        scheduler.init(12);

        let mut per_task = [0usize; 5];
        while let Some(task_index) = scheduler.current_task_index() {
            let taken = scheduler.take(usize::MAX);
            if taken == 0 {
                break;
            }
            per_task[task_index] += taken;
        }

        assert_eq!(per_task, [3, 3, 2, 2, 2]);
        assert!(scheduler.is_done());
        assert_eq!(scheduler.scheduled_tokens, 12);
    }

    #[test]
    fn init_with_zero_tokens_is_immediately_done() {
        let mut scheduler = SliceScheduler::new(4);
        scheduler.init(0);

        assert!(scheduler.is_done());
        assert_eq!(scheduler.current_task_index(), None);
        assert_eq!(scheduler.scheduled_tokens, 0);
        assert_eq!(scheduler.remaining_tokens(), 0);
    }

    #[test]
    fn schedule_for_sequence_splits_a_long_sequence_evenly() {
        let mut scheduler = SliceScheduler::new(3);
        scheduler.init(11);

        let mut slice_list = make_slices(3);
        let mut token_count = 0usize;

        scheduler.schedule_for_sequence(7, 40, 11, 5, &mut slice_list, &mut token_count);

        assert_eq!(token_count, 11);
        assert!(scheduler.is_done());
        assert_eq!(scheduler.remaining_tokens(), 0);

        assert_eq!(slice_list[0].len(), 1);
        assert_eq!(slice_list[1].len(), 1);
        assert_eq!(slice_list[2].len(), 1);

        let first = &slice_list[0][0];
        assert_eq!(first.batch_index, 7);
        assert_eq!(first.sequence_index, 40);
        assert_eq!(first.token_start_index, 5);
        assert_eq!(first.length, 4);
        assert!(!first.last_token_flag);

        let second = &slice_list[1][0];
        assert_eq!(second.batch_index, 7);
        assert_eq!(second.sequence_index, 44);
        assert_eq!(second.token_start_index, 9);
        assert_eq!(second.length, 4);
        assert!(!second.last_token_flag);

        let third = &slice_list[2][0];
        assert_eq!(third.batch_index, 7);
        assert_eq!(third.sequence_index, 48);
        assert_eq!(third.token_start_index, 13);
        assert_eq!(third.length, 3);
        assert!(!third.last_token_flag);
    }

    #[test]
    fn schedule_for_sequence_truncates_to_the_available_budget() {
        let mut scheduler = SliceScheduler::new(4);
        scheduler.init(10);

        let mut slice_list = make_slices(4);
        let mut token_count = 0usize;

        scheduler.schedule_for_sequence(3, 100, 13, 0, &mut slice_list, &mut token_count);

        assert_eq!(token_count, 10);
        assert!(scheduler.is_done());
        assert_eq!(scheduler.remaining_tokens(), 0);

        assert_eq!(slice_list[0].len(), 1);
        assert_eq!(slice_list[1].len(), 1);
        assert_eq!(slice_list[2].len(), 1);
        assert_eq!(slice_list[3].len(), 1);

        let slice0 = &slice_list[0][0];
        assert_eq!(slice0.batch_index, 3);
        assert_eq!(slice0.sequence_index, 100);
        assert_eq!(slice0.token_start_index, 0);
        assert_eq!(slice0.length, 3);

        let slice1 = &slice_list[1][0];
        assert_eq!(slice1.batch_index, 3);
        assert_eq!(slice1.sequence_index, 103);
        assert_eq!(slice1.token_start_index, 3);
        assert_eq!(slice1.length, 3);

        let slice2 = &slice_list[2][0];
        assert_eq!(slice2.batch_index, 3);
        assert_eq!(slice2.sequence_index, 106);
        assert_eq!(slice2.token_start_index, 6);
        assert_eq!(slice2.length, 2);

        let slice3 = &slice_list[3][0];
        assert_eq!(slice3.batch_index, 3);
        assert_eq!(slice3.sequence_index, 108);
        assert_eq!(slice3.token_start_index, 8);
        assert_eq!(slice3.length, 2);
    }

    #[test]
    fn schedule_for_sequence_with_zero_budget_emits_no_slices() {
        let mut scheduler = SliceScheduler::new(3);
        scheduler.init(0);

        let mut slice_list = make_slices(3);
        let mut token_count = 0usize;

        scheduler.schedule_for_sequence(1, 8, 5, 12, &mut slice_list, &mut token_count);

        assert_eq!(token_count, 0);
        assert!(scheduler.is_done());
        assert_eq!(scheduler.remaining_tokens(), 0);
        assert!(slice_list.iter().all(Vec::is_empty));
    }
}
