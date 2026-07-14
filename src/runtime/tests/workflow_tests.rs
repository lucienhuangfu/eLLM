use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;

use crate::runtime::scheduler::ScheduleTask;
use crate::runtime::scheduler::{BatchMode, PlanBuilder};
use crate::runtime::session::SessionHandle;
use crate::runtime::state::core::SlotState;
use crate::runtime::state::sequence::{DecodeList, SequenceSlice};
use crate::runtime::state::types::Phase;

#[test]
fn test_complete_slot_lifecycle() {
    let mut state = SlotState::new_start_state();

    assert_eq!(state.phase, Phase::Start);
    assert!(!state.is_active());
    assert!(state.is_available());

    state.transition_to_prefill(100, 50).unwrap();
    assert_eq!(state.phase, Phase::Prefill);
    assert!(state.is_active());
    assert!(!state.is_available());
    assert_eq!(state.sequence_index, 100);
    assert_eq!(state.kv_index, 100);
    assert_eq!(state.filling_length, 50);

    let phase_change = state.advance_sequence(30);
    assert!(phase_change.is_none());
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 20);
    assert_eq!(state.sequence_index, 130);

    let phase_change = state.advance_sequence(20);
    assert_eq!(phase_change, Some(Phase::Decode));
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.filling_length, 0);
    assert_eq!(state.sequence_index, 150);

    for _ in 0..10 {
        state.advance_sequence(1);
    }
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.sequence_index, 160);

    state.transition_to_eos().unwrap();
    assert_eq!(state.phase, Phase::Eos);
    assert!(!state.is_active());
    assert!(state.is_available());

    state.reset_to_start();
    assert_eq!(state.phase, Phase::Start);
    assert_eq!(state.sequence_index, usize::MAX);
    assert_eq!(state.kv_index, usize::MAX);
}

#[test]
fn test_scheduler_under_mixed_load() {
    let builder = PlanBuilder::new(16, 512, 4);
    let mut batch_list = Vec::new();

    for i in 0..5 {
        let mut state = SlotState::new_decode_state(i, i);
        state.phase = Phase::Decode;
        batch_list.push(state);
    }

    for i in 0..3 {
        batch_list.push(SlotState::new_prefill_state(10 + i * 100, 50));
    }

    batch_list.push(SlotState::new_start_state());
    batch_list.push(SlotState::new_start_state());

    let mut task = ScheduleTask::new(0);
    builder.build_task(&batch_list, &mut task, 1);

    assert_eq!(task.mode, BatchMode::Mixed);
    assert_eq!(task.decode_size, 5);
    assert!(task.prefill_size > 0);
}

#[test]
fn test_scheduler_with_decode_limit() {
    let builder = PlanBuilder::new(3, 1024, 4);
    let mut batch_list = Vec::new();

    for i in 0..10 {
        let mut state = SlotState::new_decode_state(i, i);
        state.phase = Phase::Decode;
        batch_list.push(state);
    }

    let mut task = ScheduleTask::new(0);
    builder.build_task(&batch_list, &mut task, 1);

    assert_eq!(task.decode_size, 3);
}

#[test]
fn test_scheduler_with_prefill_limit() {
    let builder = PlanBuilder::new(32, 100, 4);
    let batch_list = vec![
        SlotState::new_prefill_state(0, 50),
        SlotState::new_prefill_state(100, 80),
    ];

    let mut task = ScheduleTask::new(0);
    builder.build_task(&batch_list, &mut task, 1);

    assert!(task.prefill_size <= 100);
}

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

    let mut task = ScheduleTask::new(1);
    task.prefill_size = 10;
    task.decode_size = 1;
    task.prefill_list = prefill_list;
    task.decode_list = decode_list;

    assert_eq!(task.prefill_size, 10);
    assert_eq!(task.decode_size, 1);
    assert_eq!(task.task_id, 1);
    assert!(!task.prefill_list.is_empty());
}

#[test]
fn test_multiple_independent_tasks() {
    let task1 = ScheduleTask::new(1);
    let task2 = ScheduleTask::new(2);
    let task3 = ScheduleTask::new(3);

    assert_ne!(task1.task_id, task2.task_id);
    assert_ne!(task2.task_id, task3.task_id);
}

#[tokio::test]
async fn test_async_basic_operations() {
    let value = async { 42 }.await;
    assert_eq!(value, 42);
}

#[tokio::test]
async fn test_slot_state_notify_async() {
    let state = SlotState::new_decode_state(0, 0);
    let notify = state.notify();

    let task = tokio::spawn(async move {
        notify.notified().await;
        true
    });

    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    state.notify().notify_one();

    let result = task.await.unwrap();
    assert!(result);
}

#[test]
fn test_concurrent_plan_building() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    let next_task_id = Arc::new(AtomicU64::new(1));
    let mut handles = Vec::new();

    for _ in 0..4 {
        let next_task_id = Arc::clone(&next_task_id);
        let handle = thread::spawn(move || {
            let task_id = next_task_id.fetch_add(1, Ordering::Relaxed);
            let builder = PlanBuilder::new(32, 1024, 4);
            let mut batch_list = Vec::new();
            for i in 0..5 {
                let mut state = SlotState::new_decode_state(i, i);
                state.phase = Phase::Decode;
                batch_list.push(state);
            }
            let mut task = ScheduleTask::new(0);
            builder.build_task(&batch_list, &mut task, task_id);
            task.task_id
        });
        handles.push(handle);
    }

    let mut task_ids = Vec::new();
    for handle in handles {
        let task_id = handle.join().unwrap();
        task_ids.push(task_id);
    }

    task_ids.sort();
    task_ids.dedup();
    assert_eq!(task_ids.len(), 4);
}

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
    assert_eq!(final_state.phase, Phase::Prefill);
}

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

#[test]
fn test_plan_to_task_pipeline() {
    let builder = PlanBuilder::new(32, 1024, 4);
    let batch_list = vec![
        SlotState::new_prefill_state(0, 100),
        SlotState::new_decode_state(1, 1),
        SlotState::new_decode_state(2, 2),
    ];

    let mut task = ScheduleTask::new(0);
    builder.build_task(&batch_list, &mut task, 1);

    assert_eq!(task.mode, BatchMode::Mixed);
    assert_eq!(task.prefill_size, 100);
    assert_eq!(task.decode_size, 2);
}

#[test]
fn test_multiple_plan_generations() {
    let mut task_ids = Vec::new();

    for round in 0..5 {
        let builder = PlanBuilder::new(16, 512, 4);
        let mut batch_list = Vec::new();
        for i in 0..round * 2 {
            let mut state = SlotState::new_decode_state(i, i);
            state.phase = Phase::Decode;
            batch_list.push(state);
        }
        let mut task = ScheduleTask::new(0);
        builder.build_task(&batch_list, &mut task, round as u64 + 1);
        task_ids.push(task.task_id);
    }

    let unique_ids: std::collections::HashSet<u64> = task_ids.iter().cloned().collect();
    assert_eq!(unique_ids.len(), task_ids.len());
}

#[test]
fn test_full_state_machine_traversal() {
    let mut state = SlotState::new_start_state();

    assert!(SlotState::can_transition(Phase::Start, Phase::Prefill));
    state.transition_to_prefill(0, 5).unwrap();

    assert!(SlotState::can_transition(Phase::Prefill, Phase::Decode));
    let change = state.advance_sequence(5);
    assert_eq!(change, Some(Phase::Decode));

    assert!(SlotState::can_transition(Phase::Decode, Phase::Eos));
    state.transition_to_eos().unwrap();

    state.reset_to_start();
    assert_eq!(state.phase, Phase::Start);
}

#[test]
fn test_state_machine_error_recovery() {
    let mut state = SlotState::new_start_state();

    let result = state.transition_to_decode();
    assert!(result.is_err());

    assert_eq!(state.phase, Phase::Start);

    state.phase = Phase::Prefill;
    state.transition_to_timeout().unwrap();
    assert_eq!(state.phase, Phase::Timeout);

    state.transition_to_prefill(0, 10).unwrap();
    assert_eq!(state.phase, Phase::Prefill);
}

#[test]
fn test_advance_sequence_partial_processing() {
    let mut state = SlotState::new_prefill_state(0, 10);

    let result = state.advance_sequence(3);
    assert_eq!(result, None);
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 7);
    assert_eq!(state.sequence_index, 3);

    let result = state.advance_sequence(5);
    assert_eq!(result, None);
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 2);
    assert_eq!(state.sequence_index, 8);

    let result = state.advance_sequence(2);
    assert_eq!(result, Some(Phase::Decode));
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.filling_length, 0);
    assert_eq!(state.sequence_index, 10);
}

#[test]
fn test_slot_state_default() {
    let state: SlotState = Default::default();
    assert_eq!(state.phase, Phase::Start);
    assert!(!state.is_active());
}

#[test]
fn test_session_handle_clone_independence() {
    let handle1 = SessionHandle::new("session-1".to_string(), 5);
    let handle2 = handle1.clone();

    assert_eq!(handle1.session_id, handle2.session_id);
    assert_eq!(handle1.slot_index, handle2.slot_index);
    assert_eq!(handle1.is_reused, handle2.is_reused);

    let mut handle1_mut = handle1;
    handle1_mut.slot_index = 10;
    assert_eq!(handle1_mut.slot_index, 10);
    assert_eq!(handle2.slot_index, 5);
}

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
    assert!(duration.as_secs() < 1);
}

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
    let mut task = ScheduleTask::new(0);
    for _ in 0..100 {
        builder.build_task(&batch_list, &mut task, 1);
    }
    let duration = start.elapsed();

    assert!(duration.as_secs() < 5);
}

#[allow(dead_code)]
fn create_test_runtime() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn test_batch_sequence_temperature() {
    use crate::runtime::state::batch::BatchSequence;

    let batch = BatchSequence::<f32>::default();
    for &temp in &batch.batch_temperature {
        assert_eq!(temp, 1.0);
    }
}