use std::thread;
use std::time::Duration;

use crate::init::record::{BatchRecord, Phase, SequenceSlice};

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
    decode_scheduler: SliceScheduler,
    prefill_scheduler: SliceScheduler,
    prefill_list: Vec<Vec<SequenceSlice>>,
    decode_list: Vec<Vec<SequenceSlice>>,
    max_prefill_size: usize,
    max_decode_size: usize,
    thread_num: usize,
}

impl BatchScheduler {
    // Test method (simplest example):
    // 1) Build a small batch list with one PrefillBegin record.
    // 2) Call schedule_batch and verify prefill/decode counts and slices.
    // Example:
    // let mut scheduler = BatchScheduler::new(8, 1, 1);
    // let mut batch = vec![BatchRecord {
    //     phase: Phase::PrefillBegin,
    //     sequence_index: 4,
    //     snapshot_sequence_index: 0,
    //     kv_index: 0,
    // }];
    // let (prefill, decode) = scheduler.schedule_batch(&mut batch);
    // assert!(prefill > 0);
    // assert_eq!(decode, 0);
    pub fn new(sequence_length: usize, batch_size: usize, thread_num: usize) -> Self {
        let build_list = || {
            (0..thread_num)
                .map(|_| Vec::with_capacity(batch_size))
                .collect::<Vec<_>>()
        };

        Self {
            thread_num,
            decode_scheduler: SliceScheduler::new(batch_size * thread_num),
            prefill_scheduler: SliceScheduler::new(batch_size * thread_num),
            prefill_list: build_list(),
            decode_list: build_list(),
            max_prefill_size: sequence_length * batch_size,
            max_decode_size: batch_size,
        }
    }

    fn schedule_decode_only(&mut self, batch_list: &[BatchRecord], decode_count: &mut usize) {
        for task in self.decode_list.iter_mut() {
            task.clear();
        }

        let total_tokens = batch_list
            .iter()
            .filter(|record| record.phase == Phase::Decode)
            .take(self.max_decode_size)
            .count();

        self.decode_scheduler.init(total_tokens);

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

    fn compute_prefill_decode_totals(batch_list: &mut [BatchRecord]) -> (usize, usize) {
        let mut prefill_total_tokens = 0usize;
        let mut decode_total_tokens = 0usize;

        for record in batch_list.iter_mut() {
            match record.phase {
                Phase::PrefillBegin => {
                    record.snapshot_sequence_index = record.sequence_index;
                    let remaining = record
                        .snapshot_sequence_index
                        .saturating_sub(record.kv_index);
                    prefill_total_tokens += remaining;
                }
                Phase::PrefillEnd => {
                    record.snapshot_sequence_index = record.sequence_index;
                    let remaining = record
                        .snapshot_sequence_index
                        .saturating_sub(record.kv_index);
                    prefill_total_tokens += remaining;
                    decode_total_tokens += 1;
                }
                _ => {}
            }
        }

        (prefill_total_tokens, decode_total_tokens)
    }

    fn schedule_prefill_and_decode(
        &mut self,
        batch_list: &mut [BatchRecord],
        prefill_count: &mut usize,
        decode_count: &mut usize,
    ) {
        for task in self.prefill_list.iter_mut() {
            task.clear();
        }
        for task in self.decode_list.iter_mut() {
            task.clear();
        }

        let (prefill_total_tokens, decode_total_tokens) =
            Self::compute_prefill_decode_totals(batch_list);

        let total_tokens = prefill_total_tokens.min(self.max_prefill_size);
        self.prefill_scheduler.init(total_tokens);

        // decode_list is cleared above, so no decode tokens are used yet.
        self.decode_scheduler
            .init(decode_total_tokens.min(self.max_decode_size));

        for (batch_index, record) in batch_list.iter_mut().enumerate() {
            if self.prefill_scheduler.is_done() {
                break;
            }

            match record.phase {
                Phase::PrefillBegin => {
                    let remaining = record
                        .snapshot_sequence_index
                        .saturating_sub(record.kv_index);
                    self.prefill_scheduler.schedule_for_sequence(
                        batch_index,
                        record.sequence_index,
                        remaining,
                        0,
                        &mut self.prefill_list,
                        prefill_count,
                    );
                }
                Phase::PrefillEnd => {
                    let remaining = record
                        .snapshot_sequence_index
                        .saturating_sub(record.kv_index);
                    self.prefill_scheduler.schedule_for_sequence(
                        batch_index,
                        record.sequence_index,
                        remaining,
                        0,
                        &mut self.prefill_list,
                        prefill_count,
                    );
                    self.decode_scheduler.schedule_for_sequence(
                        batch_index,
                        record.sequence_index.saturating_sub(1),
                        1,
                        0,
                        &mut self.decode_list,
                        decode_count,
                    );
                }
                _ => {}
            }
        }
    }

    pub fn schedule_batch(&mut self, batch_list: &mut Vec<BatchRecord>) -> (usize, usize) {
        let decode_task_count = self.thread_num.max(1).min(self.decode_list.len().max(1));
        let prefill_task_count = self.thread_num.max(1).min(self.prefill_list.len().max(1));

        self.decode_scheduler.set_task_count(decode_task_count);
        self.prefill_scheduler.set_task_count(prefill_task_count);

        let mut prefill_count = 0usize;
        let mut decode_count = 0usize;

        loop {
            let has_decode = batch_list
                .iter()
                .any(|record| record.phase == Phase::Decode);

            if has_decode {
                self.schedule_decode_only(batch_list, &mut decode_count);
                return (prefill_count, decode_count);
            }

            let has_prefill = batch_list.iter().any(|record| {
                record.phase == Phase::PrefillBegin || record.phase == Phase::PrefillEnd
            });

            if has_prefill {
                self.schedule_prefill_and_decode(batch_list, &mut prefill_count, &mut decode_count);
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

    #[test]
    fn schedule_prefill_only() {
        // Input:
        // - Eight BatchRecord in Phase::PrefillBegin
        // - sequence_index=6, kv_index=0 => 6 tokens remaining per record
        // - sequence_length=8, batch_size=32, thread_num=8
        // Output (expected):
        // - prefill == 48 (all 8 * 6 tokens scheduled)
        // - decode == 0 (no decode tokens scheduled in prefill-only case)
        // - prefill_list[0..8] each has one slice of length 6 with token_start_index 0,6,12..42
        // - decode_list is empty across all tasks
        let mut scheduler = BatchScheduler::new(8, 32, 8);
        let mut batch = (0..8)
            .map(|_| BatchRecord {
                phase: Phase::PrefillBegin,
                sequence_index: 6,
                snapshot_sequence_index: 0,
                kv_index: 0,
                prompt_length: 0,
                notify: Notify::new(),
            })
            .collect::<Vec<_>>();

        let (prefill, decode) = scheduler.schedule_batch(&mut batch);

        assert_eq!(prefill, 48);
        assert_eq!(decode, 0);

        assert_eq!(scheduler.prefill_list.len(), 8);
        assert_eq!(scheduler.decode_list.len(), 8);

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
}
