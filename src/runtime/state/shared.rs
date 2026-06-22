use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;

use super::types::SequenceState;
use crate::runtime::executor::sync::BatchTracker;
use crate::runtime::plan::{BatchPlan, PlanBuilder};

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
    pub plan_builder: Arc<PlanBuilder>,
}

impl SharedState {
    pub fn new(
        batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SequenceState>>>,
        max_decode: usize,
        max_prefill: usize,
        thread_num: usize,
    ) -> Self {
        Self {
            batch_list,
            request_count: AtomicUsize::new(0),
            current_batch: AtomicPtr::new(std::ptr::null_mut()),
            batch_ready: AtomicBool::new(false),
            scheduler_state: AtomicUsize::new(SchedulerState::Idle as usize),
            spin_lock: AtomicBool::new(false),
            batch_tracker: BatchTracker::new(),
            plan_builder: Arc::new(PlanBuilder::new(max_decode, max_prefill, thread_num)),
        }
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
