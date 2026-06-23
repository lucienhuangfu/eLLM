use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::{Condvar, Mutex};

use super::core::SlotState;
use crate::runtime::scheduler::ScheduleTask;

pub struct SharedState {
    pub batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
    pub has_work: AtomicBool,
    pub last_task: Mutex<Option<ScheduleTask>>,
    pub work_available: Condvar,
    pub work_mutex: Mutex<bool>,
    pub active_threads: AtomicUsize,
}

impl SharedState {
    pub fn new(
        batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
    ) -> Self {
        Self {
            batch_list,
            has_work: AtomicBool::new(false),
            last_task: Mutex::new(None),
            work_available: Condvar::new(),
            work_mutex: Mutex::new(false),
            active_threads: AtomicUsize::new(0),
        }
    }

    pub fn push_request(&self) {}
}
