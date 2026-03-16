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
