use std::sync::Arc;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduler::{BatchMode, Scheduler};
use crate::runtime::session::{Phase, SlotState};

/// End-to-end phase lifecycle: Start -> Prefill -> Decode -> Eos -> Prefill again
#[test]
fn test_phase_lifecycle_integration() {
    let mut state = SlotState::new_start_state();
    assert_eq!(state.phase, Phase::Start);

    state = SlotState::new_prefill_state(0, 10);
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 10);

    state.sequence_index += 10;
    state.filling_length = 0;
    state.phase = Phase::Decode;
    assert_eq!(state.phase, Phase::Decode);

    state.phase = Phase::Eos;
    assert_eq!(state.phase, Phase::Eos);

    state = SlotState::new_prefill_state(5, 20);
    assert_eq!(state.phase, Phase::Prefill);
}

/// Scheduler builds correct task from mixed batch
#[test]
fn test_scheduler_mixed_batch() {
    let batch_list = vec![
        {
            let mut s = SlotState::new_decode_state(0, 0);
            s.phase = Phase::Decode;
            s
        },
        SlotState::new_prefill_state(10, 50),
        SlotState::new_start_state(),
    ];

    let scheduler = Scheduler::new(32, 1024, 4, Arc::new(SharedMut::new(batch_list)));
    scheduler.schedule_batch();

    scheduler.with_task(|task| {
        assert_eq!(task.mode, BatchMode::Mixed);
        assert!(task.decode_size > 0);
        assert!(task.prefill_size > 0);
    });
}

/// Scheduler respects decode limit
#[test]
fn test_scheduler_respects_decode_limit() {
    let batch_list: Vec<SlotState> = (0..20)
        .map(|i| {
            let mut s = SlotState::new_decode_state(i, i);
            s.phase = Phase::Decode;
            s
        })
        .collect();

    let scheduler = Scheduler::new(5, 1024, 4, Arc::new(SharedMut::new(batch_list)));
    scheduler.schedule_batch();

    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 5);
    });
}

/// Scheduler work flag tracks task state
#[test]
fn test_scheduler_work_tracking() {
    let batch_list = Arc::new(SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(8, 512, 1, batch_list);

    assert!(!scheduler.has_work());

    scheduler.with_task_mut(|task| {
        task.prefill_size = 1;
    });
    assert!(scheduler.has_work());

    scheduler.with_task_mut(|task| {
        task.reset();
    });
    assert!(!scheduler.has_work());
}
