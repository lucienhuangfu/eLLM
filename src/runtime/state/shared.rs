use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use super::types::SequenceState;
use crate::runtime::executor::plan::BatchPlan;
use crate::runtime::executor::tracker::BatchTracker;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SchedulerState {
    Idle = 0,
    Scheduling = 1,
    Executing = 2,
    Completing = 3,
}

pub struct SharedState {
    pub batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SequenceState>>>,
    pub request_count: AtomicUsize,
    pub current_batch: AtomicPtr<BatchPlan>,
    pub batch_ready: AtomicBool,
    pub scheduler_state: AtomicUsize,
    pub spin_lock: AtomicBool,
    pub batch_tracker: BatchTracker,
    pub plan_builder: PlanBuilderInner,
}

impl SharedState {
    pub fn new(
        batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SequenceState>>>,
    ) -> Self {
        Self {
            batch_list,
            request_count: AtomicUsize::new(0),
            current_batch: AtomicPtr::new(std::ptr::null_mut()),
            batch_ready: AtomicBool::new(false),
            scheduler_state: AtomicUsize::new(SchedulerState::Idle as usize),
            spin_lock: AtomicBool::new(false),
            batch_tracker: BatchTracker::new(),
            plan_builder: PlanBuilderInner::new(),
        }
    }

    pub fn set_plan_builder(&self, max_decode: usize, max_prefill: usize, thread_num: usize) {
        self.plan_builder
            .max_decode_size
            .store(max_decode, Ordering::Release);
        self.plan_builder
            .max_prefill_size
            .store(max_prefill, Ordering::Release);
        self.plan_builder
            .thread_num
            .store(thread_num, Ordering::Release);
    }

    pub fn push_request(&self) {
        while self
            .spin_lock
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            std::hint::spin_loop();
        }
        self.request_count.fetch_add(1, Ordering::Release);
        self.spin_lock.store(false, Ordering::Release);
    }

    pub fn take_requests(&self) -> usize {
        while self
            .spin_lock
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            std::hint::spin_loop();
        }
        let count = self.request_count.swap(0, Ordering::AcqRel);
        self.spin_lock.store(false, Ordering::Release);
        count
    }

    pub fn set_scheduler_state(&self, state: SchedulerState) {
        self.scheduler_state
            .store(state as usize, Ordering::Release);
    }

    pub fn get_scheduler_state(&self) -> SchedulerState {
        match self.scheduler_state.load(Ordering::Acquire) {
            0 => SchedulerState::Idle,
            1 => SchedulerState::Scheduling,
            2 => SchedulerState::Executing,
            3 => SchedulerState::Completing,
            _ => SchedulerState::Idle,
        }
    }

    pub fn publish_batch(&self, plan: Box<BatchPlan>) {
        let ptr = Box::into_raw(plan);
        let old = self.current_batch.swap(ptr, Ordering::Release);
        if !old.is_null() {
            unsafe { drop(Box::from_raw(old)) };
        }
        self.batch_ready.store(true, Ordering::Release);
    }

    pub fn take_batch(&self) -> Option<Box<BatchPlan>> {
        if !self.batch_ready.load(Ordering::Acquire) {
            return None;
        }
        let ptr = self.current_batch.load(Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }
        unsafe { Some(Box::from_raw(ptr)) }
    }

    pub fn clear_batch(&self) {
        self.batch_ready.store(false, Ordering::Release);
        let ptr = self
            .current_batch
            .swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null() {
            unsafe { drop(Box::from_raw(ptr)) };
        }
    }
}

pub struct PlanBuilderInner {
    max_decode_size: AtomicUsize,
    max_prefill_size: AtomicUsize,
    thread_num: AtomicUsize,
    next_task_id: AtomicU64,
}

impl PlanBuilderInner {
    pub fn new() -> Self {
        Self {
            max_decode_size: AtomicUsize::new(1),
            max_prefill_size: AtomicUsize::new(1),
            thread_num: AtomicUsize::new(1),
            next_task_id: AtomicU64::new(1),
        }
    }

    pub fn build_plan(
        &self,
        batch_list: &[SequenceState],
    ) -> crate::runtime::executor::plan::BatchPlan {
        let mut plan = crate::runtime::executor::plan::BatchPlan::new(
            self.next_task_id.fetch_add(1, Ordering::Relaxed),
        );

        let max_decode_size = self.max_decode_size.load(Ordering::Acquire);
        let max_prefill_size = self.max_prefill_size.load(Ordering::Acquire);
        let thread_num = self.thread_num.load(Ordering::Acquire);

        let mut decode_candidates: Vec<(usize, usize)> = Vec::with_capacity(max_decode_size);
        let mut prefill_candidates: Vec<(usize, usize, usize)> = Vec::new();
        let mut has_decode = false;

        for (batch_index, record) in batch_list.iter().enumerate() {
            match record.phase {
                crate::runtime::state::types::Phase::Decode => {
                    has_decode = true;
                    if decode_candidates.len() < max_decode_size {
                        decode_candidates.push((batch_index, record.sequence_index));
                    }
                }
                crate::runtime::state::types::Phase::Prefill => {
                    prefill_candidates.push((
                        batch_index,
                        record.sequence_index,
                        record.filling_length,
                    ));
                }
                _ => {}
            }
        }

        let has_prefill = !prefill_candidates.is_empty();

        if has_prefill && has_decode {
            plan.mode = crate::runtime::executor::plan::BatchMode::Mixed;
        } else if has_prefill {
            plan.mode = crate::runtime::executor::plan::BatchMode::Prefill;
        } else if has_decode {
            plan.mode = crate::runtime::executor::plan::BatchMode::Decode;
        } else {
            return plan;
        }

        if has_prefill {
            Self::build_prefill(&mut plan, &prefill_candidates, max_prefill_size, thread_num);
        }

        if has_decode {
            Self::build_decode(&mut plan, &decode_candidates);
        }

        plan
    }

    fn build_decode(
        plan: &mut crate::runtime::executor::plan::BatchPlan,
        candidates: &[(usize, usize)],
    ) {
        plan.decode_list.clear();

        for (idx, (batch_index, sequence_index)) in candidates.iter().enumerate() {
            plan.decode_list
                .push(crate::runtime::state::sequence::SequenceSlice {
                    batch_index: *batch_index,
                    sequence_index: *sequence_index,
                    token_start_index: idx,
                    length: 1,
                    last_token_flag: true,
                });
        }

        plan.decode_size = candidates.len();
    }

    fn build_prefill(
        plan: &mut crate::runtime::executor::plan::BatchPlan,
        candidates: &[(usize, usize, usize)],
        max_prefill_size: usize,
        thread_num: usize,
    ) {
        let total_tokens: usize = candidates.iter().map(|c| c.2).sum();
        let total_tokens = total_tokens.min(max_prefill_size);

        plan.prefill_list.resize_with(thread_num, || Vec::new());

        let mut prefill_count = 0usize;
        let task_count = thread_num.min(plan.prefill_list.len());

        let mut scheduler = SliceScheduler::new(task_count);
        scheduler.init(total_tokens);

        for &(batch_index, sequence_index, remaining) in candidates {
            if scheduler.is_done() {
                break;
            }

            let attention_length = remaining.min(scheduler.remaining_tokens());
            if attention_length > 0 {
                plan.decode_list
                    .push(crate::runtime::state::sequence::SequenceSlice {
                        batch_index,
                        sequence_index,
                        token_start_index: prefill_count,
                        length: attention_length,
                        last_token_flag: attention_length == remaining,
                    });
            }

            scheduler.schedule_for_sequence(
                batch_index,
                sequence_index,
                remaining,
                0,
                &mut plan.prefill_list,
                &mut prefill_count,
            );
        }

        plan.prefill_size = prefill_count;
    }
}

struct SliceScheduler {
    task_count: usize,
    total_tokens: usize,
    scheduled_tokens: usize,
    current_task: usize,
    current_task_remaining: usize,
}

impl SliceScheduler {
    fn new(task_count: usize) -> Self {
        Self {
            task_count,
            total_tokens: 0,
            scheduled_tokens: 0,
            current_task: 0,
            current_task_remaining: 0,
        }
    }

    fn init(&mut self, total_tokens: usize) {
        self.total_tokens = total_tokens;
        self.scheduled_tokens = 0;
        self.current_task = 0;
        self.current_task_remaining = if total_tokens > 0 {
            self.quota_for(0)
        } else {
            0
        };
    }

    fn is_done(&self) -> bool {
        self.scheduled_tokens >= self.total_tokens
    }

    fn remaining_tokens(&self) -> usize {
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

    fn current_task_index(&mut self) -> Option<usize> {
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

    fn take(&mut self, max_take: usize) -> usize {
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

    fn schedule_for_sequence(
        &mut self,
        batch_index: usize,
        sequence_index: usize,
        mut remaining: usize,
        token_offset: usize,
        slice_list: &mut Vec<Vec<crate::runtime::state::sequence::SequenceSlice>>,
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

            slice_list[task_index].push(crate::runtime::state::sequence::SequenceSlice {
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
