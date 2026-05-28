use super::state::SequenceSlice;

pub struct FairTaskAllocator {
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
    pub fn new(task_count: usize) -> Self {
        Self {
            task_count,
            total_tokens: 0,
            scheduled_tokens: 0,
            task_index: 0,
            task_remaining: 0,
            base_quota: 0,
            extra_quota: 0,
            active_task_count: 0,
        }
    }

    pub fn set_task_count(&mut self, task_count: usize) {
        self.task_count = task_count;
    }

    pub fn init(&mut self, total_tokens: usize) {
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

    pub fn is_done(&self) -> bool {
        self.scheduled_tokens >= self.total_tokens
    }

    pub fn scheduled_tokens(&self) -> usize {
        self.scheduled_tokens
    }

    pub fn remaining_tokens(&self) -> usize {
        self.total_tokens.saturating_sub(self.scheduled_tokens)
    }

    fn quota_for(&self, task_index: usize) -> usize {
        if task_index < self.extra_quota {
            self.base_quota + 1
        } else {
            self.base_quota
        }
    }

    pub fn current_task_index(&mut self) -> Option<usize> {
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

    pub fn take(&mut self, max_take: usize) -> usize {
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

pub struct SliceScheduler {
    allocator: FairTaskAllocator,
}

impl SliceScheduler {
    pub fn new(task_count: usize) -> Self {
        Self {
            allocator: FairTaskAllocator::new(task_count),
        }
    }

    pub fn set_task_count(&mut self, task_count: usize) {
        self.allocator.set_task_count(task_count);
    }

    pub fn init(&mut self, total_tokens: usize) {
        self.allocator.init(total_tokens);
    }

    pub fn is_done(&self) -> bool {
        self.allocator.is_done()
    }

    pub fn remaining_tokens(&self) -> usize {
        self.allocator.remaining_tokens()
    }

    pub fn schedule_for_sequence(
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