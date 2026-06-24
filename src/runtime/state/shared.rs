use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use super::core::SlotState;
use crate::runtime::scheduler::ScheduleTask;

pub struct SharedState {
    pub batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
    pub has_work: AtomicBool,
    pub current_task: UnsafeCell<ScheduleTask>,
    pub active_threads: AtomicUsize,
}

unsafe impl Sync for SharedState {}

impl SharedState {
    pub fn new(
        batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
    ) -> Self {
        Self {
            batch_list,
            has_work: AtomicBool::new(false),
            current_task: UnsafeCell::new(ScheduleTask::new(0, 0, Vec::new(), super::sequence::DecodeList::with_capacity(0), 0)),
            active_threads: AtomicUsize::new(0),
        }
    }

    pub fn push_request(&self) {}

    #[inline]
    pub fn set_task(&self, task: ScheduleTask) {
        unsafe {
            *self.current_task.get() = task;
        }
        self.has_work.store(true, Ordering::Release);
    }

    #[inline]
    pub fn get_task(&self) -> &ScheduleTask {
        unsafe { &*self.current_task.get() }
    }

    #[inline]
    pub fn clear_work(&self) {
        self.has_work.store(false, Ordering::Release);
    }
}
