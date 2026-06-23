use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::strategy::{DefaultSchedulerStrategy, SchedulerStrategy};
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::plan::BatchPlan;
use crate::runtime::state::core::SlotState;
use crate::runtime::state::shared::SharedState;

/// Scheduler 配置容器，提供调度策略和 batch 配置
/// 注意：实际调度逻辑在 ExecutorPool 中执行
pub struct Scheduler {
    batch_list: Arc<SharedMut<Vec<SlotState>>>,
    strategy: Box<dyn SchedulerStrategy>,
    thread_num: AtomicUsize,
    shared_state: Arc<SharedState>,
}

impl Scheduler {
    pub fn new(
        _sequence_length: usize,
        batch_size: usize,
        thread_num: usize,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
    ) -> Self {
        Self::build(
            batch_size,
            _sequence_length * batch_size,
            thread_num,
            batch_list,
            None,
            Box::new(DefaultSchedulerStrategy::new(
                batch_size,
                _sequence_length * batch_size,
                thread_num,
            )),
        )
    }

    pub fn with_mode(
        _sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
    ) -> Self {
        Self::build(
            batch_size,
            chunk_size,
            thread_num,
            batch_list,
            None,
            Box::new(DefaultSchedulerStrategy::new(
                batch_size, chunk_size, thread_num,
            )),
        )
    }

    pub fn with_strategy(
        _sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
        strategy: Box<dyn SchedulerStrategy>,
    ) -> Self {
        Self::build(
            batch_size, chunk_size, thread_num, batch_list, None, strategy,
        )
    }

    pub fn with_shared_state(
        _sequence_length: usize,
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
        shared_state: Arc<SharedState>,
    ) -> Self {
        Self::build(
            batch_size,
            chunk_size,
            thread_num,
            batch_list,
            Some(shared_state),
            Box::new(DefaultSchedulerStrategy::new(
                batch_size, chunk_size, thread_num,
            )),
        )
    }

    fn build(
        batch_size: usize,
        chunk_size: usize,
        thread_num: usize,
        batch_list: Arc<SharedMut<Vec<SlotState>>>,
        shared_state: Option<Arc<SharedState>>,
        strategy: Box<dyn SchedulerStrategy>,
    ) -> Self {
        let shared_state =
            shared_state.unwrap_or_else(|| Arc::new(SharedState::new(Arc::clone(&batch_list))));
        Self {
            batch_list,
            thread_num: AtomicUsize::new(thread_num),
            strategy,
            shared_state,
        }
    }

    pub fn thread_num(&self) -> usize {
        self.thread_num.load(Ordering::Acquire)
    }

    pub fn set_thread_num(&self, thread_num: usize) {
        self.thread_num.store(thread_num.max(1), Ordering::Release);
    }

    pub fn batch_list(&self) -> Arc<SharedMut<Vec<SlotState>>> {
        Arc::clone(&self.batch_list)
    }

    pub fn shared_state(&self) -> Arc<SharedState> {
        Arc::clone(&self.shared_state)
    }

    pub fn schedule_batch(&self) -> Option<BatchPlan> {
        self.batch_list.with(|batch_list| {
            let plan = self.strategy.plan_next_round(
                batch_list,
                self.thread_num.load(Ordering::Acquire),
                0,
            );
            if plan.is_empty() {
                None
            } else {
                Some(plan)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::scheduler::strategy::DefaultSchedulerStrategy;

    fn decode_state(sequence_index: usize, kv_index: usize) -> SlotState {
        SlotState::new_decode_state(sequence_index, kv_index)
    }

    fn prefill_state(sequence_index: usize, filling_length: usize) -> SlotState {
        SlotState::new_prefill_state(sequence_index, filling_length)
    }

    #[test]
    fn schedule_batch_returns_none_for_empty_batch() {
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(16, 4, 3, batch_list);

        assert!(scheduler.schedule_batch().is_none());
    }

    #[test]
    fn schedule_batch_returns_plan_for_decode() {
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(16, 4, 3, batch_list);
        scheduler.batch_list.with_mut(|batch_list| {
            batch_list.push(decode_state(100, 128));
        });

        let plan = scheduler.schedule_batch().unwrap();
        assert_eq!(plan.prefill_size, 0);
        assert_eq!(plan.decode_size, 1);
    }

    #[test]
    fn set_thread_num_updates_thread_count() {
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(16, 4, 6, batch_list);

        scheduler.set_thread_num(3);
        assert_eq!(scheduler.thread_num(), 3);

        scheduler.set_thread_num(5);
        assert_eq!(scheduler.thread_num(), 5);
    }

    #[test]
    fn custom_strategy_can_be_used() {
        let batch_list = Arc::new(SharedMut::new(Vec::new()));

        let strategy = Box::new(DefaultSchedulerStrategy::new(4, 32, 2));
        let scheduler = Scheduler::with_strategy(16, 4, 32, 2, batch_list, strategy);

        assert_eq!(scheduler.thread_num(), 2);
    }
}
