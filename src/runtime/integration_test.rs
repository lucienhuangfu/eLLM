use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;

use crate::operators::fake_echo::FakeEcho;
use crate::operators::operator::Operator;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::state::types::Phase;
use crate::runtime::{
    BatchSequence, ExecutorPool, ScheduleTask, Scheduler, SessionMode, SharedState, SlotManager,
    SlotState,
};

fn create_test_batch_sequences(
    batch_size: usize,
    sequence_length: usize,
) -> Arc<SharedMut<BatchSequence<f16>>> {
    let mut sequences = vec![0usize; batch_size * sequence_length];
    let ptr = sequences.as_mut_ptr();
    std::mem::forget(sequences);

    let mut batch_sequences = BatchSequence::<f16>::default();
    batch_sequences.sequences = ptr;
    batch_sequences.row_size = batch_size;
    batch_sequences.col_size = sequence_length;

    Arc::new(SharedMut::new(batch_sequences))
}

fn create_test_slot_manager(
    batch_size: usize,
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
) -> Arc<SlotManager<f16>> {
    Arc::new(SlotManager::new(
        batch_size,
        batch_sequences,
        SessionMode::NonReusable,
        600000,
    ))
}

fn create_test_shared_state(
    batch_list: Arc<SharedMut<Vec<SlotState>>>,
    max_decode: usize,
    max_prefill: usize,
    thread_num: usize,
) -> Arc<SharedState> {
    let (schedule_tx, _) = tokio::sync::broadcast::channel(16);
    Arc::new(SharedState::new(
        batch_list,
        max_decode,
        max_prefill,
        thread_num,
        schedule_tx,
    ))
}

fn create_test_executor_pool(
    shared_state: Arc<SharedState>,
    thread_num: usize,
) -> ExecutorPool<f16> {
    let operator_queue = vec![Operator::FakeEcho(FakeEcho)];
    ExecutorPool::new(operator_queue, shared_state, thread_num)
}

#[test]
fn test_runtime_integration_single_prefill_request() {
    let batch_size = 4;
    let sequence_length = 256;
    let thread_num = 1;

    let batch_sequences = create_test_batch_sequences(batch_size, sequence_length);
    let slot_manager = create_test_slot_manager(batch_size, batch_sequences);

    let batch_list = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect(),
    ));
    let shared_state =
        create_test_shared_state(Arc::clone(&batch_list), batch_size, 64, thread_num);

    let (broadcast_sender, _) = broadcast::channel(16);
    let scheduler = Arc::new(Scheduler::new(
        sequence_length,
        batch_size,
        thread_num,
        1,
        Duration::from_millis(10),
        broadcast_sender.clone(),
        Arc::clone(&batch_list),
        slot_manager,
    ));

    let executor_pool = create_test_executor_pool(Arc::clone(&shared_state), thread_num);

    let session_id = "test_session_1";
    let filling_length = 10;

    batch_list.with_mut(|list| {
        let entry = &mut list[0];
        crate::runtime::state::machine::SlotStateMachine::transition_to_prefill(
            entry,
            0,
            filling_length,
        )
        .unwrap();
        entry.session_id = Some(session_id.to_string());
    });

    let plan = scheduler.schedule_batch();
    assert!(plan.is_some());
    let plan = plan.unwrap();

    assert_eq!(plan.prefill_size, filling_length);
    assert_eq!(plan.prefill_list.len(), thread_num);
    assert_eq!(plan.decode_size, 0);

    let task = ScheduleTask::new(
        plan.prefill_size,
        plan.decode_size,
        plan.prefill_list,
        plan.decode_list,
        plan.task_id,
    );

    executor_pool.execute_single_thread_batch(&task);

    batch_list.with(|list| {
        let entry = &list[0];
        assert_eq!(entry.phase, Phase::Eos);
        assert_eq!(entry.kv_index, filling_length);
        assert_eq!(entry.filling_length, 0);
    });

    println!("✓ Single prefill request integration test passed");
}

#[test]
fn test_runtime_integration_multiple_prefill_requests() {
    let batch_size = 4;
    let sequence_length = 256;
    let thread_num = 2;

    let batch_sequences = create_test_batch_sequences(batch_size, sequence_length);
    let slot_manager = create_test_slot_manager(batch_size, batch_sequences);

    let batch_list = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect(),
    ));
    let shared_state =
        create_test_shared_state(Arc::clone(&batch_list), batch_size, 128, thread_num);

    let (broadcast_sender, _) = broadcast::channel(16);
    let scheduler = Arc::new(Scheduler::new(
        sequence_length,
        batch_size,
        thread_num,
        1,
        Duration::from_millis(10),
        broadcast_sender.clone(),
        Arc::clone(&batch_list),
        slot_manager,
    ));

    let executor_pool = create_test_executor_pool(Arc::clone(&shared_state), thread_num);

    let requests = vec![
        ("session_1", 0, 15),
        ("session_2", 1, 20),
        ("session_3", 2, 10),
    ];

    batch_list.with_mut(|list| {
        for (session_id, slot_idx, filling_length) in &requests {
            let entry = &mut list[*slot_idx];
            crate::runtime::state::machine::SlotStateMachine::transition_to_prefill(
                entry,
                *slot_idx * sequence_length,
                *filling_length,
            )
            .unwrap();
            entry.session_id = Some(session_id.to_string());
        }
    });

    let plan = scheduler.schedule_batch();
    assert!(plan.is_some());
    let plan = plan.unwrap();

    assert_eq!(plan.prefill_size, 45);
    assert_eq!(plan.prefill_list.len(), thread_num);
    assert_eq!(plan.decode_size, 0);

    let task = ScheduleTask::new(
        plan.prefill_size,
        plan.decode_size,
        plan.prefill_list,
        plan.decode_list,
        plan.task_id,
    );

    executor_pool.execute_single_thread_batch(&task);

    batch_list.with(|list| {
        for (_, slot_idx, filling_length) in &requests {
            let entry = &list[*slot_idx];
            assert_eq!(
                entry.phase,
                Phase::Eos,
                "Slot {} should be in Eos",
                slot_idx
            );
            assert_eq!(
                entry.kv_index, *filling_length,
                "Slot {} kv_index mismatch",
                slot_idx
            );
            assert_eq!(
                entry.filling_length, 0,
                "Slot {} filling_length should be 0",
                slot_idx
            );
        }
    });

    println!("✓ Multiple prefill requests integration test passed");
}

#[test]
fn test_runtime_integration_prefill_and_decode_mixed() {
    let batch_size = 4;
    let sequence_length = 256;
    let thread_num = 1;

    let batch_sequences = create_test_batch_sequences(batch_size, sequence_length);
    let slot_manager = create_test_slot_manager(batch_size, batch_sequences);

    let batch_list = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect(),
    ));
    let shared_state =
        create_test_shared_state(Arc::clone(&batch_list), batch_size, 64, thread_num);

    let (broadcast_sender, _) = broadcast::channel(16);
    let scheduler = Arc::new(Scheduler::new(
        sequence_length,
        batch_size,
        thread_num,
        1,
        Duration::from_millis(10),
        broadcast_sender.clone(),
        Arc::clone(&batch_list),
        slot_manager,
    ));

    let executor_pool = create_test_executor_pool(Arc::clone(&shared_state), thread_num);

    batch_list.with_mut(|list| {
        let prefill_entry = &mut list[0];
        crate::runtime::state::machine::SlotStateMachine::transition_to_prefill(
            prefill_entry,
            0,
            10,
        )
        .unwrap();
        prefill_entry.session_id = Some("prefill_session".to_string());

        let mut decode_state = SlotState::new_decode_state(sequence_length, sequence_length);
        decode_state.session_id = Some("decode_session".to_string());
        list[1] = decode_state;
    });

    let plan = scheduler.schedule_batch();
    assert!(plan.is_some());
    let plan = plan.unwrap();

    assert_eq!(plan.prefill_size, 10);
    assert_eq!(plan.decode_size, 1);
    assert_eq!(plan.prefill_list.len(), thread_num);

    let task = ScheduleTask::new(
        plan.prefill_size,
        plan.decode_size,
        plan.prefill_list,
        plan.decode_list,
        plan.task_id,
    );

    executor_pool.execute_single_thread_batch(&task);

    batch_list.with(|list| {
        let prefill_entry = &list[0];
        assert_eq!(prefill_entry.phase, Phase::Eos);

        let decode_entry = &list[1];
        assert_eq!(decode_entry.phase, Phase::Decode);
        assert_eq!(decode_entry.sequence_index, sequence_length + 1);
    });

    println!("✓ Mixed prefill and decode integration test passed");
}

#[test]
fn test_runtime_integration_scheduler_trigger() {
    let batch_size = 4;
    let sequence_length = 256;
    let thread_num = 1;

    let batch_sequences = create_test_batch_sequences(batch_size, sequence_length);
    let slot_manager = create_test_slot_manager(batch_size, batch_sequences);

    let batch_list = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect(),
    ));
    let shared_state =
        create_test_shared_state(Arc::clone(&batch_list), batch_size, 64, thread_num);

    let (broadcast_sender, mut broadcast_receiver) = broadcast::channel(16);
    let scheduler = Arc::new(Scheduler::new(
        sequence_length,
        batch_size,
        thread_num,
        1,
        Duration::from_millis(10),
        broadcast_sender,
        Arc::clone(&batch_list),
        slot_manager,
    ));

    let executor_pool = create_test_executor_pool(Arc::clone(&shared_state), thread_num);

    let session_id = "trigger_test_session";
    let slot_index = 0;
    let filling_length = 8;

    batch_list.with_mut(|list| {
        let entry = &mut list[slot_index];
        crate::runtime::state::machine::SlotStateMachine::transition_to_prefill(
            entry,
            0,
            filling_length,
        )
        .unwrap();
        entry.session_id = Some(session_id.to_string());
    });

    let task = scheduler.schedule_batch().unwrap();

    assert_eq!(task.prefill_size, filling_length);
    assert_eq!(task.decode_size, 0);

    let schedule_task = ScheduleTask::new(
        task.prefill_size,
        task.decode_size,
        task.prefill_list,
        task.decode_list,
        task.task_id,
    );

    executor_pool.execute_single_thread_batch(&schedule_task);

    batch_list.with(|list| {
        let entry = &list[slot_index];
        assert_eq!(entry.phase, Phase::Eos);
    });

    println!("✓ Scheduler trigger integration test passed");
}

#[test]
fn test_runtime_integration_full_lifecycle() {
    let batch_size = 4;
    let sequence_length = 256;
    let thread_num = 1;

    let batch_sequences = create_test_batch_sequences(batch_size, sequence_length);
    let slot_manager = create_test_slot_manager(batch_size, batch_sequences);

    let batch_list = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect(),
    ));
    let shared_state =
        create_test_shared_state(Arc::clone(&batch_list), batch_size, 64, thread_num);

    let (broadcast_sender, _) = broadcast::channel(16);
    let scheduler = Arc::new(Scheduler::new(
        sequence_length,
        batch_size,
        thread_num,
        1,
        Duration::from_millis(10),
        broadcast_sender,
        Arc::clone(&batch_list),
        slot_manager,
    ));

    let executor_pool = create_test_executor_pool(Arc::clone(&shared_state), thread_num);

    let session_id = "full_lifecycle_test";
    let filling_length = 12;

    batch_list.with_mut(|list| {
        let entry = &mut list[0];
        crate::runtime::state::machine::SlotStateMachine::transition_to_prefill(
            entry,
            0,
            filling_length,
        )
        .unwrap();
        entry.session_id = Some(session_id.to_string());
    });

    let plan = scheduler.schedule_batch().unwrap();

    let task = ScheduleTask::new(
        plan.prefill_size,
        plan.decode_size,
        plan.prefill_list,
        plan.decode_list,
        plan.task_id,
    );

    executor_pool.execute_single_thread_batch(&task);

    batch_list.with(|list| {
        let entry = &list[0];
        assert_eq!(entry.phase, Phase::Eos);
        assert_eq!(entry.kv_index, filling_length);
    });

    batch_list.with_mut(|list| {
        list[0] = SlotState::new_start_state();
        list[0].token_count = filling_length;
    });

    batch_list.with(|list| {
        let entry = &list[0];
        assert!(entry.is_available());
        assert_eq!(entry.token_count, filling_length);
    });

    println!("✓ Full lifecycle integration test passed");
}
