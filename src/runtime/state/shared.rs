use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use super::core::SlotState;
use crate::runtime::scheduler::ScheduleTask;

pub struct SharedState {
    pub batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
    pub task_in_flight: AtomicBool,
    pub last_task: Mutex<Option<ScheduleTask>>,
}

impl SharedState {
    pub fn new(
        batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
    ) -> Self {
        Self {
            batch_list,
            task_in_flight: AtomicBool::new(false),
            last_task: Mutex::new(None),
        }
    }

    pub fn push_request(&self) {}
}
