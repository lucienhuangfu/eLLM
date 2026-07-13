//! Runtime 模块集成测试
//!
//! 这些测试验证 runtime 模块各个组件之间的交互和集成行为。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::{SlotError, SlotResult};
use crate::runtime::scheduler::ScheduleTask;
use crate::runtime::scheduler::{BatchMode, BatchPlan, PlanBuilder};
use crate::runtime::session::{SessionHandle, SessionMode, SlotManager};
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::core::{SlotState, TransitionError};
use crate::runtime::state::sequence::{DecodeList, DecodeLookupResult, SequenceSlice};
use crate::runtime::state::shared::SharedState;
use crate::runtime::state::types::Phase;
use crate::runtime::ExecutorPool;

/// ========================================
/// Phase 状态机集成测试
/// ========================================

#[test]
fn test_phase_lifecycle_integration() {
    // 测试 Phase 在完整生命周期中的转换
    let mut state = SlotState::new_start_state();
    assert_eq!(state.phase, Phase::Start);

    // Start -> Prefill
    state.transition_to_prefill(0, 10).unwrap();
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 10);

    // Prefill -> Decode (通过 advance_sequence)
    let phase_changed = state.advance_sequence(10);
    assert_eq!(phase_changed, Some(Phase::Decode));
    assert_eq!(state.phase, Phase::Decode);

    // Decode -> Eos
    state.transition_to_eos().unwrap();
    assert_eq!(state.phase, Phase::Eos);

    // Eos 可以回到 Prefill
    state.transition_to_prefill(5, 20).unwrap();
    assert_eq!(state.phase, Phase::Prefill);
}

#[test]
fn test_phase_timeout_recovery() {
    let mut state = SlotState::new_prefill_state(0, 100);
    assert_eq!(state.phase, Phase::Prefill);

    // Prefill -> Timeout
    state.transition_to_timeout().unwrap();
    assert_eq!(state.phase, Phase::Timeout);

    // Timeout -> Prefill (恢复)
    state.transition_to_prefill(10, 50).unwrap();
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.sequence_index, 10);
    assert_eq!(state.filling_length, 50);
}

#[test]
fn test_invalid_phase_transitions() {
    let mut state = SlotState::new_start_state();

    // Start 不能直接到 Decode
    let result = state.transition_to_decode();
    assert!(matches!(result, Err(TransitionError::InvalidTransition)));

    // Start 不能直接到 Eos
    let result = state.transition_to_eos();
    assert!(matches!(result, Err(TransitionError::InvalidTransition)));

    // Decode 不能到 Prefill (只能往前)
    state.phase = Phase::Decode;
    let result = state.transition_to_prefill(0, 10);
    assert!(matches!(result, Err(TransitionError::InvalidTransition)));
}

#[test]
fn test_can_transition_validates_all_combinations() {
    // 有效转换
    assert!(SlotState::can_transition(Phase::Start, Phase::Prefill));
    assert!(SlotState::can_transition(Phase::Eos, Phase::Prefill));
    assert!(SlotState::can_transition(Phase::Timeout, Phase::Prefill));
    assert!(SlotState::can_transition(Phase::Prefill, Phase::Decode));
    assert!(SlotState::can_transition(Phase::Decode, Phase::Eos));
    assert!(SlotState::can_transition(Phase::Prefill, Phase::Eos));
    assert!(SlotState::can_transition(Phase::Decode, Phase::Timeout));
    assert!(SlotState::can_transition(Phase::Prefill, Phase::Timeout));

    // 无效转换
    assert!(!SlotState::can_transition(Phase::Start, Phase::Decode));
    assert!(!SlotState::can_transition(Phase::Decode, Phase::Prefill));
    assert!(!SlotState::can_transition(Phase::Eos, Phase::Decode));
    assert!(!SlotState::can_transition(Phase::Start, Phase::Eos));
}

/// ========================================
/// SequenceSlice 和 DecodeList 集成测试
/// ========================================

#[test]
fn test_decode_list_sequence_operations() {
    let mut list = DecodeList::with_capacity(10);

    // 模拟两个序列的 decode
    list.push(SequenceSlice {
        token_start_index: 0,
        batch_index: 0,
        sequence_index: 0,
        length: 6,
        last_token_flag: false,
    });
    list.push(SequenceSlice {
        token_start_index: 6,
        batch_index: 1,
        sequence_index: 0,
        length: 4,
        last_token_flag: true,
    });

    assert_eq!(list.len(), 2);
    assert_eq!(list.total_token_count(), 10);

    // 测试 lookup_global_index
    let result = list.lookup_global_index(0);
    assert_eq!(
        result,
        Some(DecodeLookupResult {
            batch_index: 0,
            sequence_index: 0,
            slice_index: 0
        })
    );

    let result = list.lookup_global_index(5);
    assert_eq!(
        result,
        Some(DecodeLookupResult {
            batch_index: 0,
            sequence_index: 5,
            slice_index: 0
        })
    );

    let result = list.lookup_global_index(6);
    assert_eq!(
        result,
        Some(DecodeLookupResult {
            batch_index: 1,
            sequence_index: 0,
            slice_index: 1
        })
    );

    let result = list.lookup_global_index(9);
    assert_eq!(
        result,
        Some(DecodeLookupResult {
            batch_index: 1,
            sequence_index: 3,
            slice_index: 1
        })
    );

    // 超出范围
    let result = list.lookup_global_index(10);
    assert!(result.is_none());
}

#[test]
fn test_decode_list_walk_global_range() {
    let mut list = DecodeList::with_capacity(3);
    list.push(SequenceSlice {
        token_start_index: 0,
        batch_index: 0,
        sequence_index: 100,
        length: 5,
        last_token_flag: false,
    });
    list.push(SequenceSlice {
        token_start_index: 5,
        batch_index: 1,
        sequence_index: 200,
        length: 5,
        last_token_flag: true,
    });

    let mut visited = Vec::new();
    list.walk_global_range(2, 8, |global_idx, batch_idx, seq_idx| {
        visited.push((global_idx, batch_idx, seq_idx));
    });

    assert_eq!(visited.len(), 6);
    // 验证从 global_index 2 到 8 的遍历
    assert_eq!(visited[0], (2, 0, 102)); // batch 0, sequence 100 + 2
    assert_eq!(visited[5], (7, 1, 202)); // batch 1, sequence 200 + 2
}

#[test]
fn test_decode_list_clear_and_reuse() {
    let mut list = DecodeList::with_capacity(2);
    list.push(SequenceSlice {
        token_start_index: 0,
        batch_index: 0,
        sequence_index: 0,
        length: 10,
        last_token_flag: true,
    });

    assert_eq!(list.len(), 1);
    list.clear();
    assert_eq!(list.len(), 0);

    // 清除后可以重新使用
    list.push(SequenceSlice {
        token_start_index: 0,
        batch_index: 5,
        sequence_index: 50,
        length: 20,
        last_token_flag: false,
    });
    assert_eq!(list.len(), 1);
    assert_eq!(list.total_token_count(), 20);
}

/// ========================================
/// BatchPlan 和 PlanBuilder 集成测试
/// ========================================

#[test]
fn test_batch_plan_empty_state() {
    let plan = BatchPlan::new(1);
    assert!(plan.is_empty());
    assert_eq!(plan.sequence_count(), 0);
    assert_eq!(plan.mode, BatchMode::Decode);
}

#[test]
fn test_plan_builder_decode_only() {
    let builder = PlanBuilder::new(32, 1024, 4);
    let mut batch_list = Vec::new();

    for i in 0..10 {
        let mut state = SlotState::new_decode_state(i, i);
        state.phase = Phase::Decode;
        batch_list.push(state);
    }

    let plan = builder.build_plan(&batch_list);
    assert_eq!(plan.mode, BatchMode::Decode);
    assert_eq!(plan.decode_size, 10);
    assert_eq!(plan.prefill_size, 0);
    assert_eq!(plan.sequence_count(), 10);
}

#[test]
fn test_plan_builder_prefill_only() {
    let builder = PlanBuilder::new(32, 1024, 4);
    let batch_list = vec![
        SlotState::new_prefill_state(0, 100),
        SlotState::new_prefill_state(1, 200),
    ];

    let plan = builder.build_plan(&batch_list);
    assert_eq!(plan.mode, BatchMode::Prefill);
    assert!(plan.prefill_size > 0);
    assert_eq!(plan.decode_size, 0);
}

#[test]
fn test_plan_builder_mixed_mode() {
    let builder = PlanBuilder::new(32, 1024, 4);
    let mut batch_list = Vec::new();

    // 添加 decode 状态
    let mut decode_state = SlotState::new_decode_state(0, 0);
    decode_state.phase = Phase::Decode;
    batch_list.push(decode_state);

    // 添加 prefill 状态
    batch_list.push(SlotState::new_prefill_state(10, 50));

    let plan = builder.build_plan(&batch_list);
    assert_eq!(plan.mode, BatchMode::Mixed);
    assert!(plan.decode_size > 0);
    assert!(plan.prefill_size > 0);
}

#[test]
fn test_plan_builder_respects_decode_limit() {
    let builder = PlanBuilder::new(5, 1024, 4); // 限制最多 5 个 decode
    let mut batch_list = Vec::new();

    for i in 0..20 {
        let mut state = SlotState::new_decode_state(i, i);
        state.phase = Phase::Decode;
        batch_list.push(state);
    }

    let plan = builder.build_plan(&batch_list);
    assert_eq!(plan.mode, BatchMode::Decode);
    assert_eq!(plan.decode_size, 5); // 限制生效
}

#[test]
fn test_plan_builder_task_id_uniqueness() {
    let builder = PlanBuilder::new(32, 1024, 4);

    let batch_list = vec![SlotState::new_prefill_state(0, 10)];
    let plan1 = builder.build_plan(&batch_list);

    let batch_list = vec![SlotState::new_prefill_state(0, 10)];
    let plan2 = builder.build_plan(&batch_list);

    // 每次调用应该产生唯一的 task_id
    assert_ne!(plan1.task_id, plan2.task_id);
}

/// ========================================
/// ScheduleTask 集成测试
/// ========================================

#[test]
fn test_schedule_task_lifecycle() {
    let prefill_list = vec![vec![SequenceSlice {
        token_start_index: 0,
        batch_index: 0,
        sequence_index: 0,
        length: 10,
        last_token_flag: false,
    }]];

    let mut decode_list = DecodeList::with_capacity(3);
    decode_list.push(SequenceSlice {
        token_start_index: 0,
        batch_index: 1,
        sequence_index: 0,
        length: 1,
        last_token_flag: true,
    });

    let task = ScheduleTask::new(10, 1, prefill_list, decode_list, 42);

    assert_eq!(task.prefill_size, 10);
    assert_eq!(task.decode_size, 1);
    assert_eq!(task.task_id, 42);
    assert!(!task.prefill_list.is_empty());
    assert!(!task.decode_list.is_empty());
}

#[test]
fn test_schedule_task_clone() {
    let prefill_list = vec![vec![SequenceSlice {
        token_start_index: 0,
        batch_index: 0,
        sequence_index: 0,
        length: 5,
        last_token_flag: false,
    }]];

    let task = ScheduleTask::new(5, 0, prefill_list, DecodeList::with_capacity(0), 1);
    let cloned_task = task.clone();

    assert_eq!(task.prefill_size, cloned_task.prefill_size);
    assert_eq!(task.decode_size, cloned_task.decode_size);
    assert_eq!(task.task_id, cloned_task.task_id);
}

/// ========================================
/// SharedState 集成测试
/// ========================================

#[test]
fn test_shared_state_basic_operations() {
    let batch_list = Arc::new(SharedMut::new(Vec::new()));
    let shared_state = SharedState::new(batch_list.clone());

    assert!(!shared_state.has_work());
    assert!(matches!(
        shared_state.current_work(),
        crate::runtime::state::shared::ExecutorWork::Idle
    ));

    let task = ScheduleTask::new(1, 0, Vec::new(), DecodeList::with_capacity(0), 9);
    shared_state.set_task(task);
    assert!(shared_state.has_work());
    assert!(matches!(
        shared_state.current_work(),
        crate::runtime::state::shared::ExecutorWork::Scheduled(_)
    ));

    shared_state.clear_work();
    assert!(!shared_state.has_work());
}

/// ========================================
/// SessionHandle 和 SessionMode 集成测试
/// ========================================

#[test]
fn test_session_handle_creation_modes() {
    let handle_new = SessionHandle::new("session-1".to_string(), 5);
    assert_eq!(handle_new.session_id, "session-1");
    assert_eq!(handle_new.slot_index, 5);
    assert!(!handle_new.is_reused);

    let handle_reused = SessionHandle::reused("session-2".to_string(), 10);
    assert_eq!(handle_reused.session_id, "session-2");
    assert_eq!(handle_reused.slot_index, 10);
    assert!(handle_reused.is_reused);
}

#[test]
fn test_session_mode_copy_and_clone() {
    let mode1 = SessionMode::Reusable;
    let mode2 = SessionMode::NonReusable;

    // Copy
    let mode1_copy = mode1;
    let mode2_copy = mode2;
    assert_eq!(mode1, mode1_copy);
    assert_eq!(mode2, mode2_copy);

    // Clone
    let mode1_clone = mode1.clone();
    let mode2_clone = mode2.clone();
    assert_eq!(mode1, mode1_clone);
    assert_eq!(mode2, mode2_clone);
}

/// ========================================
/// SlotState 状态管理集成测试
/// ========================================

#[test]
fn test_slot_state_activity_status() {
    let start_state = SlotState::new_start_state();
    assert!(!start_state.is_active());
    assert!(start_state.is_available());

    let prefill_state = SlotState::new_prefill_state(0, 10);
    assert!(prefill_state.is_active());
    assert!(!prefill_state.is_available());

    let decode_state = SlotState::new_decode_state(0, 0);
    assert!(decode_state.is_active());
    assert!(!decode_state.is_available());
}

#[test]
fn test_slot_state_touch_updates_timestamp() {
    let mut state = SlotState::new_decode_state(0, 0);
    let original_access = state.last_accessed;

    std::thread::sleep(Duration::from_millis(1));
    state.touch();

    assert!(state.last_accessed > original_access);
}

#[test]
fn test_slot_state_notify_clone() {
    let state = SlotState::new_decode_state(0, 0);
    let notify1 = state.notify();
    let notify2 = state.notify();

    // 两个 Arc 应该指向同一个 Notify 对象
    assert!(Arc::ptr_eq(&notify1, &notify2));
}

/// ========================================
/// 错误类型集成测试
/// ========================================

#[test]
fn test_slot_error_display_and_debug() {
    let err_alloc = SlotError::AllocatorUnavailable;
    assert_eq!(err_alloc.to_string(), "Slot allocator unavailable");
    assert!(format!("{:?}", err_alloc).contains("AllocatorUnavailable"));

    let err_empty = SlotError::SlotQueueEmpty;
    assert_eq!(
        err_empty.to_string(),
        "Slot queue empty while permit acquired"
    );
    assert!(format!("{:?}", err_empty).contains("SlotQueueEmpty"));

    let err_not_found = SlotError::SlotNotFound;
    assert_eq!(err_not_found.to_string(), "Slot not found");
    assert!(format!("{:?}", err_not_found).contains("SlotNotFound"));
}

#[test]
fn test_slot_result_error_propagation() {
    fn return_error() -> SlotResult<()> {
        Err(SlotError::SlotNotFound)
    }

    fn return_ok() -> SlotResult<()> {
        Ok(())
    }

    assert!(return_error().is_err());
    assert!(return_ok().is_ok());

    let result = return_error();
    assert!(matches!(result, Err(SlotError::SlotNotFound)));
}

/// ========================================
/// 端到端工作流测试
/// ========================================

#[test]
fn test_full_decode_workflow() {
    // 模拟完整的 decode 工作流
    let mut state = SlotState::new_start_state();

    // 1. 开始 prefill
    state.transition_to_prefill(0, 100).unwrap();
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 100);

    // 2. 逐步消耗 prefill tokens
    for step in 0..10 {
        let phase_changed = state.advance_sequence(10);
        if step < 9 {
            assert!(phase_changed.is_none());
            assert_eq!(state.phase, Phase::Prefill);
        } else {
            assert_eq!(phase_changed, Some(Phase::Decode));
            assert_eq!(state.phase, Phase::Decode);
        }
    }

    // 3. 执行 decode steps
    for _ in 0..5 {
        state.sequence_index += 1;
        state.token_count += 1;
    }

    // 4. 结束
    state.transition_to_eos().unwrap();
    assert_eq!(state.phase, Phase::Eos);
    assert!(!state.is_active());
}

#[test]
fn test_multiple_sequences_decode_list_operations() {
    // 模拟多个序列的 decode list
    let mut list = DecodeList::with_capacity(10);

    // 序列 0: 3 个 tokens
    list.push(SequenceSlice {
        token_start_index: 0,
        batch_index: 0,
        sequence_index: 0,
        length: 3,
        last_token_flag: false,
    });

    // 序列 1: 5 个 tokens
    list.push(SequenceSlice {
        token_start_index: 3,
        batch_index: 1,
        sequence_index: 0,
        length: 5,
        last_token_flag: true,
    });

    // 序列 2: 2 个 tokens
    list.push(SequenceSlice {
        token_start_index: 8,
        batch_index: 2,
        sequence_index: 0,
        length: 2,
        last_token_flag: false,
    });

    assert_eq!(list.total_token_count(), 10);

    // 验证各个 token 的查找
    assert!(list.lookup_global_index(0).is_some());
    assert!(list.lookup_global_index(2).is_some());
    assert!(list.lookup_global_index(3).is_some());
    assert!(list.lookup_global_index(7).is_some());
    assert!(list.lookup_global_index(9).is_some());
    assert!(list.lookup_global_index(10).is_none()); // 超出范围
}

#[test]
fn test_plan_with_mixed_active_and_inactive_states() {
    let builder = PlanBuilder::new(32, 1024, 4);
    let batch_list = vec![
        SlotState::new_start_state(),        // 非活跃
        SlotState::new_prefill_state(0, 10), // 活跃 - prefill
        SlotState::new_decode_state(1, 1),   // 活跃 - decode
        SlotState::new_start_state(),        // 非活跃
        SlotState::new_decode_state(2, 2),   // 活跃 - decode
    ];

    let plan = builder.build_plan(&batch_list);

    // 应该只包含活跃状态
    assert!(plan.decode_size >= 2 || plan.prefill_size > 0);
}

#[test]
fn test_session_lifecycle_with_slot_allocation() {
    let session_id = "test-session-123";

    // 模拟 SessionHandle 的创建和复用标记
    let handle1 = SessionHandle::new(session_id.to_string(), 0);
    assert!(!handle1.is_reused);
    assert_eq!(handle1.session_id, session_id);

    // 模拟会话复用
    let handle2 = SessionHandle::reused(session_id.to_string(), 0);
    assert!(handle2.is_reused);
    assert_eq!(handle2.session_id, session_id);
}

/// ========================================
/// 并发场景模拟测试
/// ========================================

#[test]
fn test_atomic_flag_operations() {
    let flag = AtomicBool::new(false);

    // 初始状态
    assert!(!flag.load(Ordering::SeqCst));

    // 设为 true
    flag.store(true, Ordering::SeqCst);
    assert!(flag.load(Ordering::SeqCst));

    // 设为 false
    flag.store(false, Ordering::SeqCst);
    assert!(!flag.load(Ordering::SeqCst));

    // compare_exchange
    let prev = flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst);
    assert!(prev.is_ok());

    let prev = flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst);
    assert!(prev.is_err()); // 已经变成 true 了
}

#[test]
fn test_duration_time_calculations() {
    let timeout = Duration::from_millis(100);
    let start = Instant::now();

    std::thread::sleep(Duration::from_millis(10));

    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(10));
    assert!(elapsed < Duration::from_secs(1));

    // timeout 检查
    assert!(elapsed < timeout);
}

/// ========================================
/// 边界条件测试
/// ========================================

#[test]
fn test_empty_batch_list() {
    let builder = PlanBuilder::new(32, 1024, 4);
    let plan = builder.build_plan(&[]);

    assert!(plan.is_empty());
    assert_eq!(plan.sequence_count(), 0);
    assert_eq!(plan.prefill_size, 0);
    assert_eq!(plan.decode_size, 0);
}

#[test]
fn test_zero_capacity_decode_list() {
    let mut list = DecodeList::with_capacity(0);
    assert_eq!(list.len(), 0);

    // 即使容量为 0，也可以添加元素
    list.push(SequenceSlice::default());
    assert_eq!(list.len(), 1);
}

#[test]
fn test_decode_lookup_boundary_conditions() {
    let mut list = DecodeList::with_capacity(2);
    list.push(SequenceSlice {
        token_start_index: 0,
        batch_index: 0,
        sequence_index: 0,
        length: 1,
        last_token_flag: true,
    });

    // 边界条件测试
    assert!(list.lookup_global_index(0).is_some());
    assert!(list.lookup_global_index(1).is_none()); // 超出范围
}

#[test]
fn test_advance_sequence_saturation() {
    let mut state = SlotState::new_prefill_state(0, 5);

    // 消耗超过 remaining 的 tokens
    let result = state.advance_sequence(100);
    assert_eq!(result, Some(Phase::Decode));
    assert_eq!(state.filling_length, 0);
}

#[test]
fn test_slot_state_max_values() {
    // 测试 usize::MAX 的处理
    let state = SlotState {
        sequence_index: usize::MAX,
        kv_index: usize::MAX,
        filling_length: usize::MAX,
        phase: Phase::Prefill,
        session_id: None,
        token_count: 0,
        created_at: Instant::now(),
        last_accessed: Instant::now(),
        notify: Arc::new(tokio::sync::Notify::new()),
        lru_prev: usize::MAX,
        lru_next: usize::MAX,
    };

    assert_eq!(state.sequence_index, usize::MAX);
    assert_eq!(state.kv_index, usize::MAX);
    assert_eq!(state.filling_length, usize::MAX);
}
