use crate::common::sequence_slice::SequenceSlice;

use super::scheduler_allocator::FairTaskAllocator;

pub(super) struct SliceScheduler {
    allocator: FairTaskAllocator,
}

impl SliceScheduler {
    pub(super) fn new(task_count: usize) -> Self {
        Self {
            allocator: FairTaskAllocator::new(task_count),
        }
    }

    pub(super) fn set_task_count(&mut self, task_count: usize) {
        self.allocator.set_task_count(task_count);
    }

    pub(super) fn init(&mut self, total_tokens: usize) {
        self.allocator.init(total_tokens);
    }

    pub(super) fn is_done(&self) -> bool {
        self.allocator.is_done()
    }

    pub(super) fn remaining_tokens(&self) -> usize {
        self.allocator.remaining_tokens()
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
            let Some(task_index) = self.allocator.current_task_index() else {
                break;
            };

            let token_start_index = token_offset + self.allocator.scheduled_tokens();
            let take = self.allocator.take(remaining);
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

#[derive(Clone, Copy)]
pub(super) enum BatchWork {
    Decode,
    Prefill,
    Idle,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_slices(task_num: usize) -> Vec<Vec<SequenceSlice>> {
        (0..task_num).map(|_| Vec::new()).collect()
    }

    #[test]
    // A single long sequence should be cut into per-task slices that follow
    // the allocator quota exactly.
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
    // If the round budget is smaller than the sequence length, the scheduler
    // must stop at the budget boundary and leave the rest for the next round.
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
    // Zero budget should leave the output lists untouched and keep the
    // allocator in the done state.
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
