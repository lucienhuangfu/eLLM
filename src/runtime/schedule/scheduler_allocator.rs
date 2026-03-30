pub(super) struct FairTaskAllocator {
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
    pub(super) fn new(task_count: usize) -> Self {
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

    pub(super) fn set_task_count(&mut self, task_count: usize) {
        self.task_count = task_count;
    }

    pub(super) fn init(&mut self, total_tokens: usize) {
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

    pub(super) fn is_done(&self) -> bool {
        self.scheduled_tokens >= self.total_tokens
    }

    pub(super) fn scheduled_tokens(&self) -> usize {
        self.scheduled_tokens
    }

    pub(super) fn remaining_tokens(&self) -> usize {
        self.total_tokens.saturating_sub(self.scheduled_tokens)
    }

    fn quota_for(&self, task_index: usize) -> usize {
        if task_index < self.extra_quota {
            self.base_quota + 1
        } else {
            self.base_quota
        }
    }

    pub(super) fn current_task_index(&mut self) -> Option<usize> {
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

    pub(super) fn take(&mut self, max_take: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::FairTaskAllocator;

    #[test]
    fn init_balances_a_long_run_across_tasks() {
        let mut allocator = FairTaskAllocator::new(6);
        allocator.init(47);

        let mut per_task = [0usize; 6];
        let mut trace = Vec::new();

        while let Some(task_index) = allocator.current_task_index() {
            let taken = allocator.take(usize::MAX);
            if taken == 0 {
                break;
            }
            per_task[task_index] += taken;
            trace.push((task_index, taken));
        }

        assert_eq!(per_task, [8, 8, 8, 8, 8, 7]);
        assert_eq!(trace, vec![(0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 7)]);
        assert!(allocator.is_done());
        assert_eq!(allocator.scheduled_tokens(), 47);
        assert_eq!(allocator.remaining_tokens(), 0);
        assert_eq!(allocator.current_task_index(), None);
    }

    #[test]
    fn init_with_more_tasks_than_tokens_activates_only_necessary_tasks() {
        let mut allocator = FairTaskAllocator::new(8);
        allocator.init(5);

        let mut trace = Vec::new();
        while let Some(task_index) = allocator.current_task_index() {
            let taken = allocator.take(1);
            if taken == 0 {
                break;
            }
            trace.push((task_index, taken));
        }

        assert_eq!(trace, vec![(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]);
        assert!(allocator.is_done());
        assert_eq!(allocator.scheduled_tokens(), 5);
        assert_eq!(allocator.remaining_tokens(), 0);
        assert_eq!(allocator.current_task_index(), None);
    }

    #[test]
    fn set_task_count_changes_the_quota_shape_before_init() {
        let mut allocator = FairTaskAllocator::new(2);
        allocator.set_task_count(5);
        allocator.init(12);

        let mut per_task = [0usize; 5];
        while let Some(task_index) = allocator.current_task_index() {
            let taken = allocator.take(usize::MAX);
            if taken == 0 {
                break;
            }
            per_task[task_index] += taken;
        }

        assert_eq!(per_task, [3, 3, 2, 2, 2]);
        assert!(allocator.is_done());
        assert_eq!(allocator.scheduled_tokens(), 12);
    }

    #[test]
    fn init_with_zero_tokens_is_immediately_done() {
        let mut allocator = FairTaskAllocator::new(4);
        allocator.init(0);

        assert!(allocator.is_done());
        assert_eq!(allocator.current_task_index(), None);
        assert_eq!(allocator.scheduled_tokens(), 0);
        assert_eq!(allocator.remaining_tokens(), 0);
    }
}
