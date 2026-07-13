use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use super::core::SlotState;
use crate::runtime::scheduler::ScheduleTask;

#[derive(Debug, Clone)]
pub enum ExecutorWork {
    Idle,
    Scheduled(Arc<ScheduleTask>),
}

impl ExecutorWork {
    #[inline]
    pub fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }

    #[inline]
    pub fn as_task(&self) -> Option<Arc<ScheduleTask>> {
        match self {
            Self::Idle => None,
            Self::Scheduled(task) => Some(Arc::clone(task)),
        }
    }
}

pub struct SharedState {
    pub batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
    work: Mutex<ExecutorWork>,
    pub active_threads: AtomicUsize,
}

impl SharedState {
    pub fn new(
        batch_list: Arc<crate::operators::send_sync_ptr::SharedMut<Vec<SlotState>>>,
    ) -> Self {
        Self {
            batch_list,
            work: Mutex::new(ExecutorWork::Idle),
            active_threads: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn set_task(&self, task: ScheduleTask) {
        let mut work = self.work.lock().unwrap();
        *work = ExecutorWork::Scheduled(Arc::new(task));
    }

    #[inline]
    pub fn current_work(&self) -> ExecutorWork {
        self.work.lock().unwrap().clone()
    }

    #[inline]
    pub fn current_task(&self) -> Option<Arc<ScheduleTask>> {
        self.work.lock().unwrap().as_task()
    }

    #[inline]
    pub fn clear_work(&self) {
        let mut work = self.work.lock().unwrap();
        *work = ExecutorWork::Idle;
    }

    #[inline]
    pub fn has_work(&self) -> bool {
        !self.work.lock().unwrap().is_idle()
    }
}
