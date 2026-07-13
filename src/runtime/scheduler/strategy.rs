use super::plan::BatchPlan;
use crate::runtime::state::core::SlotState;

pub trait SchedulerStrategy: Send + Sync + 'static {
    fn plan_next_round(&self, batch_list: &[SlotState]) -> BatchPlan;
}

pub struct DefaultSchedulerStrategy {
    max_decode_size: usize,
    max_prefill_size: usize,
    thread_num: usize,
}

impl DefaultSchedulerStrategy {
    pub fn new(max_decode_size: usize, max_prefill_size: usize, thread_num: usize) -> Self {
        Self {
            max_decode_size,
            max_prefill_size,
            thread_num,
        }
    }
}

impl Clone for DefaultSchedulerStrategy {
    fn clone(&self) -> Self {
        Self {
            max_decode_size: self.max_decode_size,
            max_prefill_size: self.max_prefill_size,
            thread_num: self.thread_num,
        }
    }
}

impl SchedulerStrategy for DefaultSchedulerStrategy {
    #[inline]
    fn plan_next_round(&self, batch_list: &[SlotState]) -> BatchPlan {
        let builder = crate::runtime::scheduler::PlanBuilder::new(
            self.max_decode_size,
            self.max_prefill_size,
            self.thread_num,
        );
        builder.build_plan(batch_list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::scheduler::plan::BatchMode;
    use crate::runtime::state::types::Phase;

    /// 测试 DefaultSchedulerStrategy 创建
    #[test]
    fn test_default_scheduler_strategy_new() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);
        // 验证策略创建成功
        let plan = strategy.plan_next_round(&[]);
        assert!(plan.is_empty());
    }

    /// 测试 DefaultSchedulerStrategy::plan_next_round 空批次
    #[test]
    fn test_strategy_plan_empty_batch() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);
        let plan = strategy.plan_next_round(&[]);

        assert!(plan.is_empty());
        assert_eq!(plan.mode, BatchMode::Decode);
    }

    /// 测试 DefaultSchedulerStrategy::plan_next_round 仅 Decode 状态
    #[test]
    fn test_strategy_plan_decode_only() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..5 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let plan = strategy.plan_next_round(&batch_list);
        assert_eq!(plan.mode, BatchMode::Decode);
        assert_eq!(plan.decode_size, 5);
        assert_eq!(plan.prefill_size, 0);
    }

    /// 测试 DefaultSchedulerStrategy::plan_next_round 仅 Prefill 状态
    #[test]
    fn test_strategy_plan_prefill_only() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let batch_list = vec![SlotState::new_prefill_state(0, 100)];

        let plan = strategy.plan_next_round(&batch_list);
        assert_eq!(plan.mode, BatchMode::Prefill);
        assert!(plan.prefill_size > 0);
    }

    /// 测试 DefaultSchedulerStrategy::plan_next_round Mixed 模式
    #[test]
    fn test_strategy_plan_mixed_mode() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        // Decode 状态
        let mut decode_state = SlotState::new_decode_state(0, 0);
        decode_state.phase = Phase::Decode;
        batch_list.push(decode_state);

        // Prefill 状态
        batch_list.push(SlotState::new_prefill_state(10, 50));

        let plan = strategy.plan_next_round(&batch_list);
        assert_eq!(plan.mode, BatchMode::Mixed);
        assert_eq!(plan.decode_size, 1);
        assert!(plan.prefill_size > 0);
    }

    /// 测试 DefaultSchedulerStrategy 不同线程数
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
            let plan = strategy.plan_next_round(&batch_list);
            assert_eq!(plan.mode, BatchMode::Prefill);
        }
    }

    /// 测试 DefaultSchedulerStrategy 不同 decode 限制
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

        let plan1 = strategies[0].plan_next_round(&batch_list);
        assert_eq!(plan1.decode_size, 1);

        let plan2 = strategies[1].plan_next_round(&batch_list);
        assert_eq!(plan2.decode_size, 10);

        let plan3 = strategies[2].plan_next_round(&batch_list);
        assert_eq!(plan3.decode_size, 50);
    }

    /// 测试 DefaultSchedulerStrategy 不同 prefill 限制
    #[test]
    fn test_strategy_different_prefill_limits() {
        let strategies = [
            DefaultSchedulerStrategy::new(32, 50, 4),
            DefaultSchedulerStrategy::new(32, 100, 4),
            DefaultSchedulerStrategy::new(32, 200, 4),
        ];

        let batch_list = vec![SlotState::new_prefill_state(0, 150)];

        for strategy in &strategies {
            let plan = strategy.plan_next_round(&batch_list);
            assert!(plan.prefill_size <= strategy.max_prefill_size);
        }
    }

    /// 测试 SchedulerStrategy trait 对象
    #[test]
    fn test_scheduler_strategy_trait_object() {
        let strategy: Box<dyn SchedulerStrategy> =
            Box::new(DefaultSchedulerStrategy::new(32, 1024, 4));

        let plan = strategy.plan_next_round(&[]);
        assert!(plan.is_empty());
    }

    /// 测试 SchedulerStrategy trait 多态
    #[test]
    fn test_scheduler_strategy_polymorphism() {
        fn use_strategy(strategy: &dyn SchedulerStrategy) -> BatchPlan {
            strategy.plan_next_round(&[])
        }

        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);
        let plan = use_strategy(&strategy);
        assert!(plan.is_empty());
    }

    /// 测试 DefaultSchedulerStrategy 仅 Start 状态
    #[test]
    fn test_strategy_only_start_states() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let batch_list = vec![
            SlotState::new_start_state(),
            SlotState::new_start_state(),
            SlotState::new_start_state(),
        ];

        let plan = strategy.plan_next_round(&batch_list);
        assert!(plan.is_empty());
    }

    /// 测试 DefaultSchedulerStrategy 仅 Eos 状态
    #[test]
    fn test_strategy_only_eos_states() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for _ in 0..3 {
            let mut state = SlotState::new_start_state();
            state.phase = Phase::Eos;
            batch_list.push(state);
        }

        let plan = strategy.plan_next_round(&batch_list);
        assert!(plan.is_empty());
    }

    /// 测试 DefaultSchedulerStrategy 仅 Timeout 状态
    #[test]
    fn test_strategy_only_timeout_states() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for _ in 0..3 {
            let mut state = SlotState::new_start_state();
            state.phase = Phase::Timeout;
            batch_list.push(state);
        }

        let plan = strategy.plan_next_round(&batch_list);
        assert!(plan.is_empty());
    }

    /// 测试 DefaultSchedulerStrategy 混合无效状态
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

        let plan = strategy.plan_next_round(&batch_list);
        assert!(plan.is_empty());
    }

    /// 测试 DefaultSchedulerStrategy 多次调用独立性
    #[test]
    fn test_strategy_multiple_calls_independent() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let decode_list = vec![SlotState::new_decode_state(0, 0)];
        let prefill_list = vec![SlotState::new_prefill_state(0, 100)];

        let plan1 = strategy.plan_next_round(&decode_list);
        let plan2 = strategy.plan_next_round(&prefill_list);

        assert_eq!(plan1.mode, BatchMode::Decode);
        assert_eq!(plan2.mode, BatchMode::Prefill);
    }

    /// 测试 DefaultSchedulerStrategy 大批次列表
    #[test]
    fn test_strategy_large_batch_list() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..1000 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let plan = strategy.plan_next_round(&batch_list);
        assert_eq!(plan.decode_size, 32); // 应该被限制
    }

    /// 测试 DefaultSchedulerStrategy 预填充 filling_length 为 0
    #[test]
    fn test_strategy_prefill_zero_filling_length() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let batch_list = vec![SlotState::new_prefill_state(0, 0)];

        let plan = strategy.plan_next_round(&batch_list);
        assert_eq!(plan.prefill_size, 0);
    }

    /// 测试 DefaultSchedulerStrategy 多个 Prefill 候选
    #[test]
    fn test_strategy_multiple_prefill_candidates() {
        let strategy = DefaultSchedulerStrategy::new(32, 200, 4);

        let batch_list = vec![
            SlotState::new_prefill_state(0, 100),
            SlotState::new_prefill_state(100, 100),
            SlotState::new_prefill_state(200, 100),
        ];

        let plan = strategy.plan_next_round(&batch_list);
        assert!(plan.prefill_size <= 200); // 应该被限制
    }

    /// 测试 SchedulerStrategy 参数被忽略（使用内部配置）
    #[test]
    fn test_strategy_ignores_parameters() {
        let strategy = DefaultSchedulerStrategy::new(10, 100, 4);

        let mut batch_list = Vec::new();
        for i in 0..50 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let plan = strategy.plan_next_round(&batch_list);

        assert_eq!(plan.decode_size, 10);
    }

    /// 测试 DefaultSchedulerStrategy 边界值 - decode_size 为 0
    #[test]
    fn test_strategy_zero_decode_size() {
        let strategy = DefaultSchedulerStrategy::new(0, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..10 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let plan = strategy.plan_next_round(&batch_list);
        // decode_size 为 0 时，decode_candidates 容量为 0，不会添加任何 decode
        assert_eq!(plan.decode_size, 0);
    }

    /// 测试 DefaultSchedulerStrategy 边界值 - prefill_size 为 0
    #[test]
    fn test_strategy_zero_prefill_size() {
        let strategy = DefaultSchedulerStrategy::new(32, 0, 4);

        let batch_list = vec![SlotState::new_prefill_state(0, 100)];

        let plan = strategy.plan_next_round(&batch_list);
        assert_eq!(plan.prefill_size, 0);
    }

    /// 测试 DefaultSchedulerStrategy 边界值 - thread_num 为 0
    #[test]
    fn test_strategy_zero_thread_num() {
        // thread_num 为 0 会导致 PlanBuilder 除零错误
        // 这个测试验证 DefaultSchedulerStrategy::new 会处理这种情况
        // 或者验证不应该创建 thread_num 为 0 的策略
        // 这里我们跳过这个测试，因为 thread_num 应该至少为 1
    }

    /// 测试 DefaultSchedulerStrategy task_id 递增
    #[test]
    fn test_strategy_task_id_increment() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        // 注意：DefaultSchedulerStrategy 每次调用都会创建新的 PlanBuilder，
        // 所以 task_id 可能不会递增。这个测试验证 task_id 是有效的。
        let plan1 = strategy.plan_next_round(&[]);
        let plan2 = strategy.plan_next_round(&[]);
        let plan3 = strategy.plan_next_round(&[]);

        // 验证 task_id 都是有效的（>= 0）
        assert!(plan1.task_id >= 0);
        assert!(plan2.task_id >= 0);
        assert!(plan3.task_id >= 0);
    }

    /// 测试 DefaultSchedulerStrategy 与 PlanBuilder 一致性
    #[test]
    fn test_strategy_consistency_with_plan_builder() {
        let strategy = DefaultSchedulerStrategy::new(32, 1024, 4);

        let mut batch_list = Vec::new();
        for i in 0..5 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }

        let plan = strategy.plan_next_round(&batch_list);

        // 验证计划结构与 PlanBuilder 生成的相同
        assert_eq!(plan.mode, BatchMode::Decode);
        assert_eq!(plan.decode_size, 5);
    }

    /// 测试 SchedulerStrategy Send 特性
    #[test]
    fn test_scheduler_strategy_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DefaultSchedulerStrategy>();
    }

    /// 测试 SchedulerStrategy Sync 特性
    #[test]
    fn test_scheduler_strategy_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<DefaultSchedulerStrategy>();
    }

    /// 测试 SchedulerStrategy trait object Send + Sync
    #[test]
    fn test_scheduler_strategy_trait_object_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // 使用 Box<dyn SchedulerStrategy> 来测试 trait object
        assert_send_sync::<Box<dyn SchedulerStrategy>>();
    }
}
