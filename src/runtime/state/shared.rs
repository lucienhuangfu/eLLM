use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use super::core::SlotState;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduler::ScheduleTask;

pub struct SharedState {
    pub batch_list: Arc<SharedMut<Vec<SlotState>>>,
    task: Arc<SharedMut<ScheduleTask>>,
    pub active_threads: AtomicUsize,
}

impl SharedState {
    pub fn new(batch_list: Arc<SharedMut<Vec<SlotState>>>) -> Self {
        Self {
            batch_list,
            task: Arc::new(SharedMut::new(ScheduleTask::new(0))),
            active_threads: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn task(&self) -> Arc<SharedMut<ScheduleTask>> {
        Arc::clone(&self.task)
    }

    #[inline]
    pub fn has_work(&self) -> bool {
        self.task.with(|task| !task.is_empty())
    }
}