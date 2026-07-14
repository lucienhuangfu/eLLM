use std::sync::atomic::{AtomicU64, Ordering};

use super::task::ScheduleTask;
use crate::runtime::state::core::SlotState;

pub trait SchedulerStrategy: Send + Sync + 'static {
    fn fill_task(&self, batch_list: &[SlotState], task: &mut ScheduleTask);
}

pub struct DefaultSchedulerStrategy {
    max_decode_size: usize,
    max_prefill_size: usize,
    thread_num: usize,
    next_task_id: AtomicU64,
}

impl DefaultSchedulerStrategy {
    pub fn new(max_decode_size: usize, max_prefill_size: usize, thread_num: usize) -> Self {
        Self {
            max_decode_size,
            max_prefill_size,
            thread_num,
            next_task_id: AtomicU64::new(1),
        }
    }

    fn next_task_id(&self) -> u64 {
        self.next_task_id.fetch_add(1, Ordering::Relaxed)
    }
}

impl Clone for DefaultSchedulerStrategy {
    fn clone(&self) -> Self {
        Self {
            max_decode_size: self.max_decode_size,
            max_prefill_size: self.max_prefill_size,
            thread_num: self.thread_num,
            next_task_id: AtomicU64::new(1),
        }
    }
}

impl SchedulerStrategy for DefaultSchedulerStrategy {
    #[inline]
    fn fill_task(&self, batch_list: &[SlotState], task: &mut ScheduleTask) {
        let builder = crate::runtime::scheduler::PlanBuilder::new(
            self.max_decode_size,
            self.max_prefill_size,
            self.thread_num,
        );
        builder.build_task(batch_list, task, self.next_task_id());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::scheduler::task::BatchMode;
    use crate::runtime::state::types::Phase;

    #[test]
    fn test_default_scheduler_strategy_new() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);
        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&[], &mut task);
        assert!(task.is_empty());
    }

    #[test]
    fn test_strategy_fill_empty_batch() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);
        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&[], &mut task);

        assert!(task.is_empty());
        assert_eq!(task.mode, BatchMode::Decode);
    }

    #[test]
    fn test_strategy_fill_decode_only() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..5 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);

        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 5);
        assert_eq!(task.prefill_size, 0);
    }

    #[test]
    fn test_strategy_fill_prefill_only() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let batch_list = vec![SlotState::new_prefill_state(0, 100)];

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);

        assert_eq!(task.mode, BatchMode::Prefill);
        assert!(task.prefill_size > 0);
    }

    #[test]
    fn test_strategy_fill_mixed_mode() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        let mut decode_state = SlotState::new_decode_state(0, 0);
        decode_state.phase = Phase::Decode;
        batch_list.push(decode_state);

        batch_list.push(SlotState::new_prefill_state(10, 50));

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);

        assert_eq!(task.mode, BatchMode::Mixed);
        assert_eq!(task.decode_size, 1);
        assert!(task.prefill_size > 0);
    }

    #[test]
    fn test_strategy_different_thread_nums() {
        let strategies = [
            DefaultSchedulerStrategy::new(32, 1024, 1),
            DefaultSchedulerStrategy::new(32, 1024, 2),
            DefaultSchedulerStrategy::new(32, 1024, 4),
            DefaultSchedulerStrategy::new(32, 1024, 8),
        ];

        let batch_list = vec![SlotState::new_prefill_state(0, 100)];

        for strategy in &strategies {
            let mut task = ScheduleTask::new(0);
            strategy.fill_task(&batch_list, &mut task);
            assert_eq!(task.mode, BatchMode::Prefill);
        }
    }

    #[test]
    fn test_strategy_different_decode_limits() {
        let strategies = [
            DefaultSchedulerStrategy::new(1, 1024, 4),
            DefaultSchedulerStrategy::new(10, 1024, 4),
            DefaultSchedulerStrategy::new(100, 1024, 4),
        ];

        let mut batch_list = Vec::new();
        for i in 0..50 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        strategies[0].fill_task(&batch_list, &mut task);
        assert_eq!(task.decode_size, 1);

        let mut task = ScheduleTask::new(0);
        strategies[1].fill_task(&batch_list, &mut task);
        assert_eq!(task.decode_size, 10);

        let mut task = ScheduleTask::new(0);
        strategies[2].fill_task(&batch_list, &mut task);
        assert_eq!(task.decode_size, 50);
    }

    #[test]
    fn test_strategy_different_prefill_limits() {
        let strategies = [
            DefaultSchedulerStrategy::new(32, 50, 4),
            DefaultSchedulerStrategy::new(32, 100, 4),
            DefaultSchedulerStrategy::new(32, 200, 4),
        ];

        let batch_list = vec![SlotState::new_prefill_state(0, 150)];

        for strategy in &strategies {
            let mut task = ScheduleTask::new(0);
            strategy.fill_task(&batch_list, &mut task);
            assert!(task.prefill_size <= strategy.max_prefill_size);
        }
    }

    #[test]
    fn test_scheduler_strategy_trait_object() {
        let strategy: Box<dyn SchedulerStrategy> =
            Box::new(DefaultSchedulerStrategy::new(32, 1024, 4));

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&[], &mut task);
        assert!(task.is_empty());
    }

    #[test]
    fn test_scheduler_strategy_polymorphism() {
        fn use_strategy(strategy: &dyn SchedulerStrategy, task: &mut ScheduleTask) {
            strategy.fill_task(&[], task);
        }

        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);
        let mut task = ScheduleTask::new(0);
        use_strategy(&strategy, &mut task);
        assert!(task.is_empty());
    }

    #[test]
    fn test_strategy_only_start_states() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let batch_list = vec![
            SlotState::new_start_state(),
            SlotState::new_start_state(),
            SlotState::new_start_state(),
        ];

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);
        assert!(task.is_empty());
    }

    #[test]
    fn test_strategy_only_eos_states() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for _ in 0..3 {
            let mut state = SlotState::new_start_state();
            state.phase = Phase::Eos;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);
        assert!(task.is_empty());
    }

    #[test]
    fn test_strategy_only_timeout_states() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for _ in 0..3 {
            let mut state = SlotState::new_start_state();
            state.phase = Phase::Timeout;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);
        assert!(task.is_empty());
    }

    #[test]
    fn test_strategy_mixed_inactive_states() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        batch_list.push(SlotState::new_start_state());

        let mut eos_state = SlotState::new_start_state();
        eos_state.phase = Phase::Eos;
        batch_list.push(eos_state);

        let mut timeout_state = SlotState::new_start_state();
        timeout_state.phase = Phase::Timeout;
        batch_list.push(timeout_state);

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);
        assert!(task.is_empty());
    }

    #[test]
    fn test_strategy_multiple_calls_independent() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let decode_list = vec![SlotState::new_decode_state(0, 0)];
        let prefill_list = vec![SlotState::new_prefill_state(0, 100)];

        let mut task1 = ScheduleTask::new(0);
        strategy.fill_task(&decode_list, &mut task1);

        let mut task2 = ScheduleTask::new(0);
        strategy.fill_task(&prefill_list, &mut task2);

        assert_eq!(task1.mode, BatchMode::Decode);
        assert_eq!(task2.mode, BatchMode::Prefill);
    }

    #[test]
    fn test_strategy_large_batch_list() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..1000 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);
        assert_eq!(task.decode_size, 32);
    }

    #[test]
    fn test_strategy_prefill_zero_filling_length() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let batch_list = vec![SlotState::new_prefill_state(0, 0)];

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);
        assert_eq!(task.prefill_size, 0);
    }

    #[test]
    fn test_strategy_multiple_prefill_candidates() {
        let strategy = DefaultSchedulerStrategy::new(32, 200, 4);

        let batch_list = vec![
            SlotState::new_prefill_state(0, 100),
            SlotState::new_prefill_state(100, 100),
            SlotState::new_prefill_state(200, 100),
        ];

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);
        assert!(task.prefill_size <= 200);
    }

    #[test]
    fn test_strategy_ignores_parameters() {
        let strategy = DefaultSchedulerStrategy::new(10, 100, 4);

        let mut batch_list = Vec::new();
        for i in 0..50 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);

        assert_eq!(task.decode_size, 10);
    }

    #[test]
    fn test_strategy_zero_decode_size() {
        let strategy = DefaultSchedulerStrategy::new(0, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..10 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);
        assert_eq!(task.decode_size, 0);
    }

    #[test]
    fn test_strategy_zero_prefill_size() {
        let strategy = DefaultSchedulerStrategy::new(32, 0, 4);

        let batch_list = vec![SlotState::new_prefill_state(0, 100)];

        let mut task = ScheduleTask::new(0);
        strategy.fill_task(&batch_list, &mut task);
        assert_eq!(task.prefill_size, 0);
    }

    #[test]
    fn test_strategy_task_id_increment() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut task1 = ScheduleTask::new(0);
        strategy.fill_task(&[], &mut task1);

        let mut task2 = ScheduleTask::new(0);
        strategy.fill_task(&[], &mut task2);

        let mut task3 = ScheduleTask::new(0);
        strategy.fill_task(&[], &mut task3);

        assert!(task1.task_id >= 0);
        assert!(task2.task_id >= 0);
        assert!(task3.task_id >= 0);
    }

    #[test]
    fn test_scheduler_strategy_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DefaultSchedulerStrategy>();
    }

    #[test]
    fn test_scheduler_strategy_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<DefaultSchedulerStrategy>();
    }

    #[test]
    fn test_scheduler_strategy_trait_object_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn SchedulerStrategy>>();
    }
}
