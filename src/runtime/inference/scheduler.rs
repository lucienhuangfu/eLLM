use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::common::send_sync_ptr::SharedMut;
use crate::runtime::inference::state::{Phase, SequenceSlice, SequenceState};

struct FairTaskAllocator {
    task_count: usize,
    total_tokens: usize,
    scheduled_tokens: usize,
    task_index: usize,
    task_remaining: usize,
    base_quota: usize,
    extra_quota: usize,
    active_task_count: usize,
}

impl FairTaskAllocator {
    fn new(task_count: usize) -> Self {
        Self {
            task_count: task_count.max(1),
            total_tokens: 0,
            scheduled_tokens: 0,
            task_index: 0,
            task_remaining: 0,
            base_quota: 0,
            extra_quota: 0,
            active_task_count: 0,
        }
    }

    fn set_task_count(&mut self, task_count: usize) {
        self.task_count = task_count.max(1);
    }

    fn init(&mut self, total_tokens: usize) {
        self.total_tokens = total_tokens;
        self.base_quota = self.total_tokens / self.task_count;
        self.extra_quota = self.total_tokens % self.task_count;
        self.active_task_count = if self.base_quota == 0 {
            self.extra_quota
        } else {
            self.task_count
        };
        self.task_index = 0;
        self.task_remaining = if self.total_tokens > 0 {
            self.quota_for(0)
        } else {
            0
        };
        self.scheduled_tokens = 0;
    }

    fn is_done(&self) -> bool {
        self.scheduled_tokens >= self.total_tokens
    }

    fn scheduled_tokens(&self) -> usize {
        self.scheduled_tokens
    }

    fn quota_for(&self, task_index: usize) -> usize {
        if task_index < self.extra_quota {
            self.base_quota + 1
        } else {
            self.base_quota
        }
    }

    fn current_task_index(&mut self) -> Option<usize> {
        if self.is_done() {
            return None;
        }

        while self.task_index < self.active_task_count && self.task_remaining == 0 {
            self.task_index += 1;
            if self.task_index < self.active_task_count {
                self.task_remaining = self.quota_for(self.task_index);
            }
        }

        if self.task_index >= self.active_task_count {
            return None;
        }

        Some(self.task_index)
    }

    fn take(&mut self, max_take: usize) -> usize {
        if self.is_done() || max_take == 0 {
            return 0;
        }

        let available = self.total_tokens - self.scheduled_tokens;
        let take = max_take.min(available).min(self.task_remaining);
        if take == 0 {
            return 0;
        }

        self.scheduled_tokens += take;
        self.task_remaining = self.task_remaining.saturating_sub(take);
        take
    }
}

struct SliceScheduler {
    allocator: FairTaskAllocator,
}

impl SliceScheduler {
    fn new(task_count: usize) -> Self {
        Self {
            allocator: FairTaskAllocator::new(task_count),
        }
    }

    fn set_task_count(&mut self, task_count: usize) {
        self.allocator.set_task_count(task_count);
    }

    fn init(&mut self, total_tokens: usize) {
        self.allocator.init(total_tokens);
    }

    fn is_done(&self) -> bool {
        self.allocator.is_done()
    }

    fn schedule_for_sequence(
        &mut self,
        batch_index: usize,
        sequence_index: usize,
        mut remaining: usize,
        token_offset: usize,
        slice_list: &mut Vec<Vec<SequenceSlice>>,
        token_count: &mut usize,
    ) -> usize {
        if self.is_done() {
            return 0;
        }

        // Schedule tokens contiguously for this sequence.
        while remaining > 0 && !self.is_done() {
            let Some(task_index) = self.allocator.current_task_index() else {
                break;
            };

            let token_start_index = token_offset + self.allocator.scheduled_tokens();
            let take = self.allocator.take(remaining);
            if take == 0 {
                break;
            }

            let task = &mut slice_list[task_index];
            let slice = SequenceSlice {
                batch_index,
                sequence_index,
                token_start_index,
                lift_index: 0,
                length: take,
            };

            task.push(slice);

            *token_count += take;
            remaining -= take;
        }

        self.allocator.scheduled_tokens()
    }
}

pub struct BatchScheduler {
    pub prefill_list: Vec<Vec<SequenceSlice>>,
    pub decode_list: Vec<Vec<SequenceSlice>>,
    pub batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    decode_scheduler: SliceScheduler,
    prefill_scheduler: SliceScheduler,
    max_prefill_size: usize,
    max_decode_size: usize,
    thread_num: usize,
}

impl BatchScheduler {
    // Test method (simplest example):
    // 1) Build a small batch list with one Prefill record.
    // 2) Call schedule_batch and verify prefill/decode counts and slices.
    // Example:
    // let mut scheduler = BatchScheduler::new(8, 1, 1);
    // scheduler.batch_list.push(SequenceState {
    //     phase: Phase::Prefill,
    //     sequence_index: 4,
    //     kv_index: 0,
    //     notify: std::sync::Arc::new(tokio::sync::Notify::new()),
    // });
    // let (prefill, decode) = scheduler.schedule_batch();
    // assert!(prefill > 0);
    // assert_eq!(decode, 0);
    pub fn new(sequence_length: usize, batch_size: usize, thread_num: usize) -> Self {
        let build_list = || {
            (0..thread_num)
                .map(|_| Vec::with_capacity(batch_size))
                .collect::<Vec<_>>()
        };

        Self {
            max_decode_size: batch_size,
            max_prefill_size: sequence_length * batch_size,
            batch_list: Arc::new(SharedMut::new(Vec::with_capacity(batch_size))),
            thread_num,
            decode_scheduler: SliceScheduler::new(batch_size),
            prefill_scheduler: SliceScheduler::new(batch_size * thread_num),
            prefill_list: build_list(),
            decode_list: build_list(),
        }
    }

    fn schedule_decode_only(&mut self, decode_count: &mut usize) {
        for task in self.decode_list.iter_mut() {
            task.clear();
        }

        let total_tokens = {
            let batch_list = unsafe { &*self.batch_list.get() };
            batch_list
                .iter()
                .filter(|record| record.phase == Phase::Decode)
                .take(self.max_decode_size)
                .count()
        };

        self.decode_scheduler.init(total_tokens);

        let batch_list = unsafe { &*self.batch_list.get() };
        for (batch_index, record) in batch_list.iter().enumerate() {
            if self.decode_scheduler.is_done() {
                break;
            }
            if record.phase != Phase::Decode {
                continue;
            }
            self.decode_scheduler.schedule_for_sequence(
                batch_index,
                record.sequence_index,
                1,
                0,
                &mut self.decode_list,
                decode_count,
            );
        }
    }

    fn schedule_prefill_and_switch_to_decode(&mut self, prefill_count: &mut usize) {
        for task in self.prefill_list.iter_mut() {
            task.clear();
        }
        for task in self.decode_list.iter_mut() {
            task.clear();
        }

        let mut prefill_total_tokens = 0usize;
        {
            let batch_list = unsafe { &*self.batch_list.get() };
            for record in batch_list.iter() {
                if record.phase == Phase::Prefill {
                    prefill_total_tokens += record.sequence_index.saturating_sub(record.kv_index);
                }
            }
        }

        let total_tokens = prefill_total_tokens.min(self.max_prefill_size);
        self.prefill_scheduler.init(total_tokens);

        // decode_list is cleared above, so no decode tokens are used yet.
        self.decode_scheduler.init(0);

        let batch_list = unsafe { &mut *self.batch_list.get() };
        for (batch_index, record) in batch_list.iter_mut().enumerate() {
            if self.prefill_scheduler.is_done() {
                break;
            }

            if record.phase == Phase::Prefill {
                let remaining = record.sequence_index.saturating_sub(record.kv_index);
                let scheduled_before = *prefill_count;
                self.prefill_scheduler.schedule_for_sequence(
                    batch_index,
                    record.sequence_index,
                    remaining,
                    0,
                    &mut self.prefill_list,
                    prefill_count,
                );

                let scheduled_for_record = prefill_count.saturating_sub(scheduled_before);
                if remaining > 0 && scheduled_for_record == remaining {
                    record.phase = Phase::Decode;
                }
            }
        }
    }

    pub fn schedule_batch(&mut self) -> (usize, usize) {
        let decode_task_count = self.thread_num.max(1).min(self.decode_list.len().max(1));
        let prefill_task_count = self.thread_num.max(1).min(self.prefill_list.len().max(1));

        self.decode_scheduler.set_task_count(decode_task_count);
        self.prefill_scheduler.set_task_count(prefill_task_count);

        let mut prefill_count = 0usize;
        let mut decode_count = 0usize;

        loop {
            let has_decode = {
                let batch_list = unsafe { &*self.batch_list.get() };
                batch_list
                    .iter()
                    .any(|record| record.phase == Phase::Decode)
            };

            if has_decode {
                self.schedule_decode_only(&mut decode_count);
                return (prefill_count, decode_count);
            }

            let has_prefill = {
                let batch_list = unsafe { &*self.batch_list.get() };
                batch_list
                    .iter()
                    .any(|record| record.phase == Phase::Prefill)
            };

            if has_prefill {
                self.schedule_prefill_and_switch_to_decode(&mut prefill_count);
                return (prefill_count, decode_count);
            }

            thread::sleep(Duration::from_millis(1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::Notify;

    fn state(phase: Phase, sequence_index: usize, kv_index: usize) -> SequenceState {
        SequenceState {
            phase,
            sequence_index,
            kv_index,
            notify: std::sync::Arc::new(Notify::new()),
        }
    }

    #[test]
    fn schedule_prefill_only() {
        // Input:
        // - Eight SequenceState in Phase::Prefill
        // - sequence_index=6, kv_index=0 => 6 tokens remaining per record
        // - sequence_length=8, batch_size=32, thread_num=8
        // Output (expected):
        // - prefill == 48 (all 8 * 6 tokens scheduled)
        // - decode == 0 (no decode tokens scheduled in prefill-only case)
        // - prefill_list[0..8] each has one slice of length 6 with token_start_index 0,6,12..42
        // - decode_list is empty across all tasks
        let batch_size = 32;
        let mut scheduler = BatchScheduler::new(8, batch_size, 8);
        {
            let batch_list = unsafe { &mut *scheduler.batch_list.get() };
            for _ in 0..8 {
                batch_list.push(state(Phase::Prefill, 6, 0));
            }
        }

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 48);
        assert_eq!(decode, 0);

        assert_eq!(scheduler.prefill_list.len(), 8);
        assert_eq!(scheduler.decode_list.len(), 8);

        let batch_list = unsafe { &*scheduler.batch_list.get() };
        for record in batch_list.iter().take(8) {
            assert_eq!(record.phase, Phase::Decode);
        }

        for task_index in 0..8 {
            assert!(scheduler.decode_list[task_index].is_empty());
        }

        for task_index in 0..8 {
            assert_eq!(scheduler.prefill_list[task_index].len(), 1);
            let slice = &scheduler.prefill_list[task_index][0];
            assert_eq!(slice.batch_index, task_index);
            assert_eq!(slice.sequence_index, 6);
            assert_eq!(slice.token_start_index, task_index * 6);
            assert_eq!(slice.length, 6);
        }
    }

    #[test]
    fn schedule_decode_only_prioritizes_decode_tasks() {
        // Input:
        // - batch_size=2 limits decode scheduling to 2 records
        // - Mixed phases: Decode, Prefill, Decode, Decode
        // Output (expected):
        // - decode == 2, prefill == 0
        // - Only first two decode records are scheduled
        // - decode slices are evenly assigned to two tasks
        // - prefill_list stays empty because decode has priority
        let mut scheduler = BatchScheduler::new(8, 2, 2);
        {
            let batch_list = unsafe { &mut *scheduler.batch_list.get() };
            batch_list.push(state(Phase::Decode, 11, 10));
            batch_list.push(state(Phase::Prefill, 6, 0));
            batch_list.push(state(Phase::Decode, 12, 11));
            batch_list.push(state(Phase::Decode, 13, 12));
        }

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 0);
        assert_eq!(decode, 2);

        for task_index in 0..2 {
            assert!(scheduler.prefill_list[task_index].is_empty());
            assert_eq!(scheduler.decode_list[task_index].len(), 1);
            assert_eq!(scheduler.decode_list[task_index][0].length, 1);
        }

        let first = &scheduler.decode_list[0][0];
        assert_eq!(first.batch_index, 0);
        assert_eq!(first.sequence_index, 11);
        assert_eq!(first.token_start_index, 0);

        let second = &scheduler.decode_list[1][0];
        assert_eq!(second.batch_index, 2);
        assert_eq!(second.sequence_index, 12);
        assert_eq!(second.token_start_index, 1);
    }

    #[test]
    fn fair_task_allocator_balances_tokens() {
        // Input:
        // - task_count=3
        // - total_tokens=11
        // Output (expected):
        // - token allocation per task = [4, 4, 3]
        // - scheduled_tokens == 11
        let mut allocator = FairTaskAllocator::new(3);
        allocator.init(11);

        let mut per_task = [0usize; 3];
        while let Some(task_index) = allocator.current_task_index() {
            let taken = allocator.take(10);
            if taken == 0 {
                break;
            }
            per_task[task_index] += taken;
        }

        assert_eq!(per_task, [4, 4, 3]);
        assert!(allocator.is_done());
        assert_eq!(allocator.scheduled_tokens(), 11);
    }

    #[test]
    fn fair_task_allocator_zero_tokens() {
        // Input:
        // - task_count=4
        // - total_tokens=0
        // Output (expected):
        // - allocator is done immediately
        // - current_task_index == None
        // - scheduled_tokens == 0
        let mut allocator = FairTaskAllocator::new(4);
        allocator.init(0);

        assert!(allocator.is_done());
        assert_eq!(allocator.current_task_index(), None);
        assert_eq!(allocator.scheduled_tokens(), 0);
    }

    #[test]
    fn schedule_prefill_partial_keeps_unscheduled_records_in_prefill() {
        // Input:
        // - sequence_length=4, batch_size=2 => max_prefill_size = 8
        // - Two Prefill records each require 6 tokens (total 12 > 8)
        // Output (expected):
        // - first record fully scheduled -> switches to Decode
        // - second record partially scheduled -> remains Prefill
        let mut scheduler = BatchScheduler::new(4, 2, 2);
        {
            let batch_list = unsafe { &mut *scheduler.batch_list.get() };
            batch_list.push(state(Phase::Prefill, 6, 0));
            batch_list.push(state(Phase::Prefill, 6, 0));
        }

        let (prefill, decode) = scheduler.schedule_batch();

        assert_eq!(prefill, 8);
        assert_eq!(decode, 0);

        let batch_list = unsafe { &*scheduler.batch_list.get() };
        assert_eq!(batch_list[0].phase, Phase::Decode);
        assert_eq!(batch_list[1].phase, Phase::Prefill);
    }
}

