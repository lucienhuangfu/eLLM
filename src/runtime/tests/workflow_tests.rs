use std::sync::Arc;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduler::{BatchMode, Scheduler};
use crate::runtime::session::{Phase, SlotState};

/// Complete slot lifecycle: Start -> Prefill -> partial Decode -> Eos -> reset -> new Prefill
#[test]
fn test_complete_slot_lifecycle() {
    let mut state = SlotState::new_start_state();
    assert!(!state.is_active());

    state.transition_to_prefill(100, 50).unwrap();
    assert!(state.is_active());
    assert_eq!(state.sequence_index, 100);
    assert_eq!(state.filling_length, 50);

    // Partial prefill
    state.advance_sequence(30);
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 20);

    // Complete prefill
    state.advance_sequence(20);
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.sequence_index, 150);

    // Decode steps
    for _ in 0..10 {
        state.advance_sequence(1);
    }
    assert_eq!(state.sequence_index, 160);

    // End and reset
    state.transition_to_eos().unwrap();
    assert!(state.is_available());

    state.reset_to_start();
    assert_eq!(state.phase, Phase::Start);
    assert_eq!(state.sequence_index, usize::MAX);
}

/// Realistic: new requests arriving during decode
#[test]
fn test_new_requests_during_decode() {
    let batch_list = Arc::new(SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(8, 512, 4, Arc::clone(&batch_list));

    // Initial decode slots
    batch_list.with_mut(|batch_list| {
        for i in 0..4 {
            let mut s = SlotState::new_decode_state(i, i);
            s.phase = Phase::Decode;
            batch_list.push(s);
        }
    });

    // Schedule a few decode rounds
    for _ in 0..3 {
        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert_eq!(task.mode, BatchMode::Decode);
            assert_eq!(task.decode_size, 4);
        });
        batch_list.with_mut(|batch_list| {
            for s in batch_list.iter_mut() {
                if s.phase == Phase::Decode {
                    s.advance_sequence(1);
                }
            }
        });
    }

    // New prefill requests arrive
    batch_list.with_mut(|batch_list| {
        batch_list.push(SlotState::new_prefill_state(100, 64));
        batch_list.push(SlotState::new_prefill_state(200, 32));
    });

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.mode, BatchMode::Mixed);
        assert_eq!(task.decode_size, 4);
        assert!(task.prefill_size > 0);
    });
}

/// Slot reuse: EOS -> reset -> new Prefill
#[test]
fn test_slot_reuse_workflow() {
    let batch_list = Arc::new(SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(4, 256, 2, Arc::clone(&batch_list));

    batch_list.with_mut(|batch_list| {
        batch_list.push(SlotState::new_decode_state(0, 0));
    });

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 1);
    });

    // End sequence
    batch_list.with_mut(|batch_list| {
        batch_list[0].transition_to_eos().unwrap();
    });
    assert!(!scheduler.schedule_batch());

    // Reuse slot
    batch_list.with_mut(|batch_list| {
        batch_list[0].reset_to_start();
        batch_list[0].transition_to_prefill(100, 50).unwrap();
    });

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.mode, BatchMode::Prefill);
        assert!(task.prefill_size > 0);
    });
}

/// Partial sequence completion: some sequences end while others continue
#[test]
fn test_partial_sequence_completion() {
    let batch_list = Arc::new(SharedMut::new(Vec::new()));
    let scheduler = Scheduler::new(8, 512, 4, Arc::clone(&batch_list));

    batch_list.with_mut(|batch_list| {
        for i in 0..5 {
            let mut s = SlotState::new_decode_state(i, i);
            s.phase = Phase::Decode;
            batch_list.push(s);
        }
    });

    // Run some decode steps, then end some sequences
    for step in 0..10 {
        let active = batch_list.with(|bl| bl.iter().filter(|s| s.phase == Phase::Decode).count());

        if active == 0 {
            break;
        }

        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, active);
        });

        if step == 2 {
            batch_list.with_mut(|bl| {
                bl[0].transition_to_eos().unwrap();
            });
        }
        if step == 4 {
            batch_list.with_mut(|bl| {
                bl[2].transition_to_eos().unwrap();
                bl[3].transition_to_eos().unwrap();
            });
        }

        batch_list.with_mut(|batch_list| {
            for s in batch_list.iter_mut() {
                if s.phase == Phase::Decode {
                    s.advance_sequence(1);
                }
            }
        });
    }

    let active = batch_list.with(|bl| bl.iter().filter(|s| s.phase == Phase::Decode).count());
    assert_eq!(active, 2);
}
