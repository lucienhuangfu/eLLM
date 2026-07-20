use std::sync::Arc;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{Phase, SlotState};

fn advance_slot(slot: &mut SlotState, steps: usize) {
    slot.sequence_index += steps;
    if slot.phase == Phase::Prefill {
        slot.filling_length = slot.filling_length.saturating_sub(steps);
        if slot.filling_length == 0 {
            slot.phase = Phase::Decode;
        }
    }
}

/// Complete slot lifecycle: Start -> Prefill -> partial Decode -> Eos -> reset -> new Prefill
#[test]
fn test_complete_slot_lifecycle() {
    let mut state = SlotState::new_start_state();

    state = SlotState::new_prefill_state(100, 50);
    assert_eq!(state.sequence_index, 100);
    assert_eq!(state.filling_length, 50);

    // Partial prefill
    advance_slot(&mut state, 30);
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 20);

    // Complete prefill
    advance_slot(&mut state, 20);
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.sequence_index, 150);

    // Decode steps
    for _ in 0..10 {
        advance_slot(&mut state, 1);
    }
    assert_eq!(state.sequence_index, 160);

    // End and reset
    state.phase = Phase::Eos;
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
            assert_eq!(task.decode_size, 4);
        });
        batch_list.with_mut(|batch_list| {
            for s in batch_list.iter_mut() {
                if s.phase == Phase::Decode {
                    advance_slot(s, 1);
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
        batch_list[0].phase = Phase::Eos;
    });
    assert!(!scheduler.schedule_batch());

    // Reuse slot
    batch_list.with_mut(|batch_list| {
        batch_list[0].reset_to_start();
        batch_list[0] = SlotState::new_prefill_state(100, 50);
    });

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
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
                bl[0].phase = Phase::Eos;
            });
        }
        if step == 4 {
            batch_list.with_mut(|bl| {
                bl[2].phase = Phase::Eos;
                bl[3].phase = Phase::Eos;
            });
        }

        batch_list.with_mut(|batch_list| {
            for s in batch_list.iter_mut() {
                if s.phase == Phase::Decode {
                    advance_slot(s, 1);
                }
            }
        });
    }

    let active = batch_list.with(|bl| bl.iter().filter(|s| s.phase == Phase::Decode).count());
    assert_eq!(active, 2);
}
