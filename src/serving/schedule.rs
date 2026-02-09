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
        }
    }

    fn set_task_count(&mut self, task_count: usize) {
        self.task_count = task_count.max(1);
    }

    fn init(&mut self, total_tokens: usize) {
        self.total_tokens = total_tokens;
        self.base_quota = self.total_tokens / self.task_count;
        self.extra_quota = self.total_tokens % self.task_count;
        self.task_index = 0;
        self.task_remaining = if self.total_tokens > 0 {
            self.base_quota + if self.extra_quota > 0 { 1 } else { 0 }
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

    fn current_task_index(&mut self) -> Option<usize> {
        if self.is_done() {
            return None;
        }

        while self.task_index < self.task_count && self.task_remaining == 0 {
            self.task_index += 1;
            if self.task_index < self.task_count {
                let has_extra = self.task_index < self.extra_quota;
                self.task_remaining = self.base_quota + if has_extra { 1 } else { 0 };
            }
        }

        if self.task_index >= self.task_count {
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

struct DecodeScheduler {
    tokens_used: usize,
    allocator: FairTaskAllocator,
}

impl DecodeScheduler {
    fn new(task_count: usize) -> Self {
        Self {
            tokens_used: 0,
            allocator: FairTaskAllocator::new(task_count),
        }
    }

    fn set_task_count(&mut self, task_count: usize) {
        self.allocator.set_task_count(task_count);
    }

    fn init(&mut self, total_tokens: usize, tokens_used: usize, max_decode_size: usize) {
        let remaining_budget = max_decode_size.saturating_sub(tokens_used);
        self.tokens_used = tokens_used;
        self.allocator.init(total_tokens.min(remaining_budget));
    }

    fn schedule(
        &mut self,
        batch_index: usize,
        sequence_index: usize,
        decode_list: &mut Vec<Vec<SequenceSlice>>,
        decode_count: &mut usize,
    ) {
        let Some(task_index) = self.allocator.current_task_index() else {
            return;
        };

        let task = &mut decode_list[task_index];
        let slice = SequenceSlice {
            batch_index,
            sequence_index,
            token_start_index: self.tokens_used + self.allocator.scheduled_tokens(),
            length: 1,
        };
        task.push(slice);

        *decode_count += 1;
        self.allocator.take(1);
    }
}

struct PrefillScheduler {
    allocator: FairTaskAllocator,
}

impl PrefillScheduler {
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

    fn schedule_for_record(
        &mut self,
        batch_index: usize,
        record: &mut BatchRecord,
        prefill_list: &mut Vec<Vec<SequenceSlice>>,
        prefill_count: &mut usize,
    ) {
        if self.is_done() {
            return;
        }

        let mut remaining = record
            .snapshot_sequence_index
            .saturating_sub(record.kv_index);

        // Schedule prefill tokens contiguously for this sequence.
        while remaining > 0 && !self.is_done() {
            let Some(task_index) = self.allocator.current_task_index() else {
                break;
            };

            let token_start_index = self.allocator.scheduled_tokens();
            let take = self.allocator.take(remaining);
            if take == 0 {
                break;
            }

            let task = &mut prefill_list[task_index];
            let slice = SequenceSlice {
                batch_index,
                sequence_index: record.sequence_index,
                token_start_index,
                length: take,
            };

            task.push(slice);

            record.kv_index = record.kv_index.saturating_add(take);
            *prefill_count += take;
            remaining -= take;
        }
    }
}

pub struct BatchScheduler {
    thread_num: usize,
    decode_scheduler: DecodeScheduler,
    prefill_scheduler: PrefillScheduler,
    prefill_list: Vec<Vec<SequenceSlice>>,
    decode_list: Vec<Vec<SequenceSlice>>,
    max_prefill_size: usize,
    max_decode_size: usize,
}

impl BatchScheduler {
    pub fn new(sequence_length: usize, batch_size: usize, thread_num: usize) -> Self {
        let build_list = || {
            (0..thread_num)
                .map(|_| Vec::with_capacity(batch_size))
                .collect::<Vec<_>>()
        };

        Self {
            thread_num,
            decode_scheduler: DecodeScheduler::new(batch_size * thread_num),
            prefill_scheduler: PrefillScheduler::new(batch_size * thread_num),
            prefill_list: build_list(),
            decode_list: build_list(),
            max_prefill_size: sequence_length * batch_size,
            max_decode_size: batch_size,
        }
    }

    pub fn schedule_batch(&mut self, batch_list: &mut Vec<BatchRecord>) -> (usize, usize) {
        let prefill_list = &mut self.prefill_list;
        let decode_list = &mut self.decode_list;
        let max_prefill_size = self.max_prefill_size;
        let max_decode_size = self.max_decode_size;

        let decode_task_count = self.thread_num.max(1).min(decode_list.len().max(1));
        let prefill_task_count = self.thread_num.max(1).min(prefill_list.len().max(1));

        self.decode_scheduler.set_task_count(decode_task_count);
        self.prefill_scheduler.set_task_count(prefill_task_count);

        let mut prefill_count = 0usize;
        let mut decode_count = 0usize;

        loop {
            let has_decode = batch_list
                .iter()
                .any(|record| record.phase == Phase::Decode);

            if has_decode {
                for task in decode_list.iter_mut() {
                    task.clear();
                }

                let total_tokens = batch_list
                    .iter()
                    .filter(|record| record.phase == Phase::Decode)
                    .count()
                    .min(max_decode_size);

                self.decode_scheduler.init(total_tokens, 0, max_decode_size);

                for (batch_index, record) in batch_list.iter().enumerate() {
                    if record.phase != Phase::Decode {
                        continue;
                    }
                    self.decode_scheduler.schedule(
                        batch_index,
                        record.sequence_index,
                        decode_list,
                        &mut decode_count,
                    );
                }
                return (prefill_count, decode_count);
            }

            let has_prefill = batch_list.iter().any(|record| {
                record.phase == Phase::PrefillBegin || record.phase == Phase::PrefillEnd
            });

            if has_prefill {
                for task in prefill_list.iter_mut() {
                    task.clear();
                }

                for task in decode_list.iter_mut() {
                    task.clear();
                }
                let mut prefill_total_tokens = 0usize;
                let mut decode_total_tokens = 0usize;

                for record in batch_list.iter_mut() {
                    match record.phase {
                        Phase::PrefillBegin => {
                            // if record.snapshot_sequence_index != record.sequence_index {
                            record.snapshot_sequence_index = record.sequence_index;
                            // }
                            let remaining = record
                                .snapshot_sequence_index
                                .saturating_sub(record.kv_index);
                            prefill_total_tokens += remaining;
                        }
                        Phase::PrefillEnd => {
                            // if record.snapshot_sequence_index != record.sequence_index {
                            record.snapshot_sequence_index = record.sequence_index;
                            // }
                            let remaining = record
                                .snapshot_sequence_index
                                .saturating_sub(record.kv_index);
                            prefill_total_tokens += remaining;
                            decode_total_tokens += 1;
                        }
                        _ => {}
                    }
                }

                let total_tokens = prefill_total_tokens.min(max_prefill_size);
                self.prefill_scheduler.init(total_tokens);

                // decode_list is cleared above, so no decode tokens are used yet.
                self.decode_scheduler
                    .init(decode_total_tokens, 0, max_decode_size);

                for (batch_index, record) in batch_list.iter_mut().enumerate() {
                    if self.prefill_scheduler.is_done() {
                        break;
                    }

                    match record.phase {
                        Phase::PrefillBegin => {
                            self.prefill_scheduler.schedule_for_record(
                                batch_index,
                                record,
                                prefill_list,
                                &mut prefill_count,
                            );
                        }
                        Phase::PrefillEnd => {
                            self.prefill_scheduler.schedule_for_record(
                                batch_index,
                                record,
                                prefill_list,
                                &mut prefill_count,
                            );
                            self.decode_scheduler.schedule(
                                batch_index,
                                record.sequence_index.saturating_sub(1),
                                decode_list,
                                &mut decode_count,
                            );
                        }
                        _ => {}
                    }
                }
                return (prefill_count, decode_count);
            }

            // if !has_decode && !has_prefill {
            thread::sleep(Duration::from_millis(1));
            // continue;
            // }
        }
    }
}
