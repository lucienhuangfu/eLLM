//! Runtime 模块端到端工作流集成测试
//!
//! 这些测试模拟 runtime 模块中的完整工作流，包括：
//! - 状态机生命周期
//! - 调度计划生成
//! - 批处理序列管理
//! - 会话和槽位分配

use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;

use crate::runtime::plan::{BatchMode, PlanBuilder};
use crate::runtime::scheduler::ScheduleTask;
use crate::runtime::session::SessionHandle;
use crate::runtime::state::core::SlotState;
use crate::runtime::state::sequence::{DecodeList, SequenceSlice};
use crate::runtime::state::types::Phase;

/// 模拟完整的 prefill -> decode -> eos 生命周期
#[test]
fn test_complete_slot_lifecycle() {
    let mut state = SlotState::new_start_state();

    // 1. 初始状态
    assert_eq!(state.phase, Phase::Start);
    assert!(!state.is_active());
    assert!(state.is_available());

    // 2. 进入 prefill
    state.transition_to_prefill(100, 50).unwrap();
    assert_eq!(state.phase, Phase::Prefill);
    assert!(state.is_active());
    assert!(!state.is_available());
    assert_eq!(state.sequence_index, 100);
    assert_eq!(state.kv_index, 100);
    assert_eq!(state.filling_length, 50);

    // 3. 逐步执行 prefill (消耗 tokens)
    let phase_change = state.advance_sequence(30);
    assert!(phase_change.is_none()); // 还在 prefill
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 20);
    assert_eq!(state.sequence_index, 130);

    // 4. 完成 prefill 进入 decode
    let phase_change = state.advance_sequence(20);
    assert_eq!(phase_change, Some(Phase::Decode));
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.filling_length, 0);
    assert_eq!(state.sequence_index, 150);

    // 5. 执行多个 decode 步骤
    for _ in 0..10 {
        state.advance_sequence(1);
    }
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.sequence_index, 160);

    // 6. 达到 EOS
    state.transition_to_eos().unwrap();
    assert_eq!(state.phase, Phase::Eos);
    assert!(!state.is_active());
    assert!(state.is_available());

    // 7. 重置到 start
    state.reset_to_start();
    assert_eq!(state.phase, Phase::Start);
    assert_eq!(state.sequence_index, usize::MAX);
    assert_eq!(state.kv_index, usize::MAX);
}

/// 测试调度器在混合负载下的行为
#[test]
fn test_scheduler_under_mixed_load() {
    let builder = PlanBuilder::new(16, 512, 4);
    let mut batch_list = Vec::new();

    // 创建混合场景
    // 5 个 decode
    for i in 0..5 {
        let mut state = SlotState::new_decode_state(i, i);
        state.phase = Phase::Decode;
        batch_list.push(state);
    }

    // 3 个 prefill
    for i in 0..3 {
        batch_list.push(SlotState::new_prefill_state(10 + i * 100, 50));
    }

    // 2 个空闲
    batch_list.push(SlotState::new_start_state());
    batch_list.push(SlotState::new_start_state());

    let plan = builder.build_plan(&batch_list);

    assert_eq!(plan.mode, BatchMode::Mixed);
    assert_eq!(plan.decode_size, 5);
    assert!(plan.prefill_size > 0);
}

/// 测试调度器在 decode 限制下的行为
#[test]
fn test_scheduler_with_decode_limit() {
    let builder = PlanBuilder::new(3, 1024, 4);
    let mut batch_list = Vec::new();

    // 创建 10 个 decode states
    for i in 0..10 {
        let mut state = SlotState::new_decode_state(i, i);
        state.phase = Phase::Decode;
        batch_list.push(state);
    }

    let plan = builder.build_plan(&batch_list);

    // decode 数量应该被限制
    assert_eq!(plan.decode_size, 3);
}

/// 测试调度器在 prefill 限制下的行为
#[test]
fn test_scheduler_with_prefill_limit() {
    let builder = PlanBuilder::new(32, 100, 4);
    let batch_list = vec![
        SlotState::new_prefill_state(0, 50),
        SlotState::new_prefill_state(100, 80),
    ];

    let plan = builder.build_plan(&batch_list);

    // prefill 应该被限制到 100
    assert!(plan.prefill_size <= 100);
}

/// 测试 ScheduleTask 的创建和消费
#[test]
fn test_schedule_task_creation_and_use() {
    let prefill_list = vec![vec![SequenceSlice {
        token_start_index: 0,
        batch_index: 0,
        sequence_index: 0,
        length: 10,
        last_token_flag: false,
    }]];

    let mut decode_list = DecodeList::with_capacity(2);
    decode_list.push(SequenceSlice {
        token_start_index: 0,
        batch_index: 1,
        sequence_index: 0,
        length: 1,
        last_token_flag: true,
    });

    let task = ScheduleTask::new(10, 1, prefill_list, decode_list, 1);

    assert_eq!(task.prefill_size, 10);
    assert_eq!(task.decode_size, 1);
    assert_eq!(task.task_id, 1);
    assert!(!task.prefill_list.is_empty());
}

/// 测试多个 schedule task 的独立性
#[test]
fn test_multiple_independent_tasks() {
    let task1 = ScheduleTask::new(10, 5, Vec::new(), DecodeList::with_capacity(0), 1);
    let task2 = ScheduleTask::new(20, 10, Vec::new(), DecodeList::with_capacity(0), 2);
    let task3 = ScheduleTask::new(30, 15, Vec::new(), DecodeList::with_capacity(0), 3);

    assert_ne!(task1.task_id, task2.task_id);
    assert_ne!(task2.task_id, task3.task_id);
    assert_eq!(task1.prefill_size, 10);
    assert_eq!(task2.prefill_size, 20);
    assert_eq!(task3.prefill_size, 30);
}

/// 异步测试：测试基本的 async 操作
#[tokio::test]
async fn test_async_basic_operations() {
    let value = async { 42 }.await;
    assert_eq!(value, 42);
}

/// 异步测试：测试 tokio 通知
#[tokio::test]
async fn test_slot_state_notify_async() {
    let state = SlotState::new_decode_state(0, 0);
    let notify = state.notify();

    // 启动一个等待通知的任务
    let task = tokio::spawn(async move {
        notify.notified().await;
        true
    });

    // 给一点时间让任务开始等待
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    // 触发通知
    state.notify().notify_one();

    // 等待任务完成
    let result = task.await.unwrap();
    assert!(result);
}

/// 测试并发场景下的 plan builder
#[test]
fn test_concurrent_plan_building() {
    use std::thread;

    let builder = Arc::new(PlanBuilder::new(32, 1024, 4));
    let mut handles = Vec::new();

    for _ in 0..4 {
        let builder = Arc::clone(&builder);
        let handle = thread::spawn(move || {
            let mut batch_list = Vec::new();
            for i in 0..5 {
                let mut state = SlotState::new_decode_state(i, i);
                state.phase = Phase::Decode;
                batch_list.push(state);
            }
            builder.build_plan(&batch_list)
        });
        handles.push(handle);
    }

    let mut task_ids = Vec::new();
    for handle in handles {
        let plan = handle.join().unwrap();
        task_ids.push(plan.task_id);
    }

    // 所有 task_id 应该唯一
    task_ids.sort();
    task_ids.dedup();
    assert_eq!(task_ids.len(), 4);
}

/// 测试状态机的并发安全转换
#[test]
fn test_concurrent_state_transitions() {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::thread;

    let state = Arc::new(Mutex::new(SlotState::new_start_state()));
    let mut handles = Vec::new();

    for _ in 0..10 {
        let state = Arc::clone(&state);
        let handle = thread::spawn(move || {
            let mut state = state.lock().unwrap();
            let _ = state.transition_to_prefill(0, 10);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_state = state.lock().unwrap();
    // 最终状态应该是 Prefill
    assert_eq!(final_state.phase, Phase::Prefill);
}

/// 测试 DecodeList 的并发访问
#[test]
fn test_concurrent_decode_list_access() {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::thread;

    let list = Arc::new(Mutex::new(DecodeList::with_capacity(100)));
    let mut handles = Vec::new();

    for i in 0..10 {
        let list = Arc::clone(&list);
        let handle = thread::spawn(move || {
            let mut list = list.lock().unwrap();
            for j in 0..10 {
                list.push(SequenceSlice {
                    token_start_index: i * 10 + j,
                    batch_index: i,
                    sequence_index: j,
                    length: 1,
                    last_token_flag: false,
                });
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let list = list.lock().unwrap();
    assert_eq!(list.len(), 100);
}

/// 测试完整的 plan -> task 流程
#[test]
fn test_plan_to_task_pipeline() {
    let builder = PlanBuilder::new(32, 1024, 4);
    let batch_list = vec![
        SlotState::new_prefill_state(0, 100),
        SlotState::new_decode_state(1, 1),
        SlotState::new_decode_state(2, 2),
    ];

    // 1. 创建 plan
    let plan = builder.build_plan(&batch_list);
    assert_eq!(plan.mode, BatchMode::Mixed);

    // 2. 转换为 ScheduleTask
    let task = ScheduleTask::new(
        plan.prefill_size,
        plan.decode_size,
        plan.prefill_list.clone(),
        plan.decode_list.clone(),
        plan.task_id,
    );

    assert_eq!(task.task_id, plan.task_id);
    assert_eq!(task.prefill_size, plan.prefill_size);
    assert_eq!(task.decode_size, plan.decode_size);
}

/// 测试多次 plan 生成
#[test]
fn test_multiple_plan_generations() {
    let builder = PlanBuilder::new(16, 512, 4);
    let mut plans = Vec::new();

    for round in 0..5 {
        let mut batch_list = Vec::new();
        for i in 0..round * 2 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }
        plans.push(builder.build_plan(&batch_list));
    }

    // 每次 plan 应该有唯一的 task_id
    let task_ids: Vec<u64> = plans.iter().map(|p| p.task_id).collect();
    let unique_ids: std::collections::HashSet<u64> = task_ids.iter().cloned().collect();
    assert_eq!(unique_ids.len(), task_ids.len());
}

/// 测试从 start 到 eos 的完整状态机
#[test]
fn test_full_state_machine_traversal() {
    let mut state = SlotState::new_start_state();

    // Start -> Prefill
    assert!(SlotState::can_transition(Phase::Start, Phase::Prefill));
    state.transition_to_prefill(0, 5).unwrap();

    // Prefill -> Decode
    assert!(SlotState::can_transition(Phase::Prefill, Phase::Decode));
    let change = state.advance_sequence(5);
    assert_eq!(change, Some(Phase::Decode));

    // Decode -> Eos
    assert!(SlotState::can_transition(Phase::Decode, Phase::Eos));
    state.transition_to_eos().unwrap();

    // Eos -> Start (reset)
    state.reset_to_start();
    assert_eq!(state.phase, Phase::Start);
}

/// 测试状态机的错误恢复
#[test]
fn test_state_machine_error_recovery() {
    let mut state = SlotState::new_start_state();

    // 尝试无效转换
    let result = state.transition_to_decode();
    assert!(result.is_err());

    // 状态应该保持不变
    assert_eq!(state.phase, Phase::Start);

    // 转换到 timeout
    state.phase = Phase::Prefill;
    state.transition_to_timeout().unwrap();
    assert_eq!(state.phase, Phase::Timeout);

    // 从 timeout 恢复
    state.transition_to_prefill(0, 10).unwrap();
    assert_eq!(state.phase, Phase::Prefill);
}

/// 测试 AdvanceSequence 的部分处理
#[test]
fn test_advance_sequence_partial_processing() {
    let mut state = SlotState::new_prefill_state(0, 10);

    // 部分处理
    let result = state.advance_sequence(3);
    assert_eq!(result, None);
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 7);
    assert_eq!(state.sequence_index, 3);

    // 继续部分处理
    let result = state.advance_sequence(5);
    assert_eq!(result, None);
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 2);
    assert_eq!(state.sequence_index, 8);

    // 最后部分
    let result = state.advance_sequence(2);
    assert_eq!(result, Some(Phase::Decode));
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.filling_length, 0);
    assert_eq!(state.sequence_index, 10);
}

/// 测试 SlotState 的默认实现
#[test]
fn test_slot_state_default() {
    let state: SlotState = Default::default();
    assert_eq!(state.phase, Phase::Start);
    assert!(!state.is_active());
}

/// 测试 SessionHandle 的 Clone 行为
#[test]
fn test_session_handle_clone_independence() {
    let handle1 = SessionHandle::new("session-1".to_string(), 5);
    let handle2 = handle1.clone();

    assert_eq!(handle1.session_id, handle2.session_id);
    assert_eq!(handle1.slot_index, handle2.slot_index);
    assert_eq!(handle1.is_reused, handle2.is_reused);

    // 修改 handle1 不应影响 handle2
    let mut handle1_mut = handle1;
    handle1_mut.slot_index = 10;
    assert_eq!(handle1_mut.slot_index, 10);
    assert_eq!(handle2.slot_index, 5);
}

/// 性能测试：大量 slot 状态管理
#[test]
fn test_performance_large_slot_management() {
    let start = Instant::now();

    let mut states = Vec::with_capacity(1000);
    for i in 0..1000 {
        let mut state = SlotState::new_prefill_state(i, 100);
        state.phase = Phase::Prefill;
        states.push(state);
    }

    let _ = states[0].advance_sequence(50);
    let _ = states[500].advance_sequence(100);
    let _ = states[999].transition_to_eos().unwrap();

    let duration = start.elapsed();
    assert!(duration.as_secs() < 1); // 应该在 1 秒内完成
}

/// 性能测试：大量 DecodeList 操作
#[test]
fn test_performance_large_decode_list() {
    let start = Instant::now();

    let mut list = DecodeList::with_capacity(10000);
    for i in 0..10000 {
        list.push(SequenceSlice {
            token_start_index: i,
            batch_index: i % 100,
            sequence_index: i,
            length: 1,
            last_token_flag: i % 2 == 0,
        });
    }

    assert_eq!(list.len(), 10000);
    let _ = list.lookup_global_index(5000);

    let duration = start.elapsed();
    assert!(duration.as_secs() < 1);
}

/// 性能测试：plan builder
#[test]
fn test_performance_plan_builder() {
    let builder = PlanBuilder::new(1024, 65536, 16);
    let mut batch_list = Vec::with_capacity(500);

    for i in 0..500 {
        let mut state = SlotState::new_decode_state(i, i);
        state.phase = Phase::Decode;
        batch_list.push(state);
    }

    let start = Instant::now();
    for _ in 0..100 {
        let _ = builder.build_plan(&batch_list);
    }
    let duration = start.elapsed();

    assert!(duration.as_secs() < 5);
}

/// 创建一个 tokio runtime 用于测试
#[allow(dead_code)]
fn create_test_runtime() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// 测试 batch_temperature 设置
#[test]
fn test_batch_sequence_temperature() {
    use crate::runtime::state::batch::BatchSequence;

    let batch = BatchSequence::<f32>::default();
    // 默认温度应该全部为 1.0
    for &temp in &batch.batch_temperature {
        assert_eq!(temp, 1.0);
    }
}
