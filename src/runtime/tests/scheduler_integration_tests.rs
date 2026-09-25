use std::sync::Arc;

use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::Phase;
use crate::runtime::session::SlotState;

use super::test_utils::*;

#[test]
fn test_empty_batch_returns_no_work() {
    let slot_list = make_slot_list(Vec::new());
    let scheduler = Scheduler::new(16, 512, 4, slot_list);
    assert!(!scheduler.schedule_batch());
    assert!(!scheduler.has_work());
}

#[test]
fn test_prefill_only_batch() {
    let slot_list = make_slot_list(vec![make_prefill_state(0, 64), make_prefill_state(100, 32)]);
    let scheduler = Scheduler::new(8, 256, 2, Arc::clone(&slot_list));

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 0);
        assert_eq!(task.prefill_size, 64 + 32);
        assert!(!task.is_empty());
    });
}

#[test]
fn test_decode_only_batch() {
    let slot_list = make_slot_list(vec![
        make_decode_state(0, 0),
        make_decode_state(10, 10),
        make_decode_state(20, 20),
    ]);
    let scheduler = Scheduler::new(8, 256, 2, Arc::clone(&slot_list));

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 3);
        assert_eq!(task.prefill_size, 0);
    });
}

#[test]
fn test_mixed_prefill_and_decode() {
    let slot_list = make_slot_list(vec![
        make_decode_state(0, 0),
        make_decode_state(10, 10),
        make_prefill_state(100, 50),
        make_prefill_state(200, 30),
    ]);
    let scheduler = Scheduler::new(8, 256, 2, Arc::clone(&slot_list));

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 2);
        assert!(task.prefill_size > 0);
    });
}

#[test]
fn test_decode_respects_max_limit() {
    let slot_list: Vec<crate::runtime::session::SlotState> = (0..20)
        .map(|i| {
            let mut s = make_decode_state(i, i);
            s.phase = Phase::Decode;
            s
        })
        .collect();

    let scheduler = Scheduler::new(5, 1024, 4, make_slot_list(slot_list));
    scheduler.schedule_batch();

    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 5);
    });
}

#[test]
fn test_chunked_prefill_across_multiple_rounds() {
    const MAX_PREFILL: usize = 100;
    const TOTAL_TOKENS: usize = 250;

    let slot_list = make_slot_list(vec![make_prefill_state(0, TOTAL_TOKENS)]);
    let scheduler = Scheduler::new(8, MAX_PREFILL, 2, Arc::clone(&slot_list));

    let mut rounds = 0;
    let mut total_prefilled = 0;

    loop {
        let has_work = scheduler.schedule_batch();
        if !has_work {
            break;
        }

        let prefill_size = scheduler.with_task(|t| t.prefill_size);
        total_prefilled += prefill_size;
        rounds += 1;

        slot_list.with_mut(|bl| {
            advance_slot(&mut bl[0], prefill_size);
        });

        if slot_list.with(|bl| bl[0].phase == Phase::Decode) {
            break;
        }
    }

    assert_eq!(total_prefilled, TOTAL_TOKENS);
    assert_eq!(rounds, 3);
}

#[test]
fn test_full_lifecycle_prefill_to_decode_to_eos() {
    const MAX_DECODE: usize = 8;
    const MAX_PREFILL: usize = 512;
    const THREADS: usize = 4;

    let slot_list = make_slot_list(Vec::new());
    let scheduler = Scheduler::new(MAX_DECODE, MAX_PREFILL, THREADS, Arc::clone(&slot_list));

    slot_list.with_mut(|bl| {
        bl.push(make_prefill_state(0, 64));
        bl.push(make_prefill_state(200, 48));
    });

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 0);
        assert_eq!(task.prefill_size, 64 + 48);
    });

    slot_list.with_mut(|bl| {
        for s in bl.iter_mut() {
            let fl = s.filling_length();
            advance_slot(s, fl);
        }
    });

    for _ in 0..5 {
        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, 2);
            assert_eq!(task.prefill_size, 0);
        });
        slot_list.with_mut(|bl| {
            for s in bl.iter_mut() {
                if s.phase == Phase::Decode {
                    advance_slot(s, 1);
                }
            }
        });
    }

    slot_list.with_mut(|bl| {
        bl[0].phase = Phase::Eos;
    });

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 1);
    });

    slot_list.with_mut(|bl| {
        bl[1].phase = Phase::Eos;
    });

    assert!(!scheduler.schedule_batch());
}

#[test]
fn test_new_prefill_arrives_during_decode() {
    let slot_list = make_slot_list(Vec::new());
    let scheduler = Scheduler::new(8, 512, 4, Arc::clone(&slot_list));

    slot_list.with_mut(|bl| {
        for i in 0..3 {
            let mut s = make_decode_state(i, i);
            s.phase = Phase::Decode;
            bl.push(s);
        }
    });

    for _ in 0..2 {
        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, 3);
        });
        slot_list.with_mut(|bl| {
            for s in bl.iter_mut() {
                if s.phase == Phase::Decode {
                    advance_slot(s, 1);
                }
            }
        });
    }

    slot_list.with_mut(|bl| {
        bl.push(make_prefill_state(100, 64));
        bl.push(make_prefill_state(200, 32));
    });

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 3);
        assert!(task.prefill_size > 0);
    });
}

#[test]
fn test_partial_sequence_completion() {
    let slot_list = make_slot_list(Vec::new());
    let scheduler = Scheduler::new(8, 512, 4, Arc::clone(&slot_list));

    slot_list.with_mut(|bl| {
        for i in 0..5 {
            let mut s = make_decode_state(i, i);
            s.phase = Phase::Decode;
            bl.push(s);
        }
    });

    for step in 0..10 {
        let active = slot_list.with(|bl| bl.iter().filter(|s| s.phase == Phase::Decode).count());
        if active == 0 {
            break;
        }

        assert!(scheduler.schedule_batch());
        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, active);
        });

        if step == 2 {
            slot_list.with_mut(|bl| bl[0].phase = Phase::Eos);
        }
        if step == 4 {
            slot_list.with_mut(|bl| {
                bl[2].phase = Phase::Eos;
                bl[3].phase = Phase::Eos;
            });
        }

        slot_list.with_mut(|bl| {
            for s in bl.iter_mut() {
                if s.phase == Phase::Decode {
                    advance_slot(s, 1);
                }
            }
        });
    }

    let active = slot_list.with(|bl| bl.iter().filter(|s| s.phase == Phase::Decode).count());
    assert_eq!(active, 2);
}

#[test]
fn test_slices_token_layout_prefill_then_decode() {
    let slot_list = make_slot_list(vec![
        make_decode_state(0, 0),
        make_decode_state(100, 100),
        make_prefill_state(300, 50),
    ]);
    let scheduler = Scheduler::new(16, 1024, 2, slot_list);

    assert!(scheduler.schedule_batch());
    let task = scheduler.with_task(|t| t.clone());

    assert_eq!(task.prefill_size, 50);
    assert_eq!(task.decode_size, 2);

    let slices = &task.slices;
    assert_eq!(slices.len(), 3);

    assert_eq!(slices[0].batch_index, 2);
    assert_eq!(slices[0].token_start_index, 0);
    assert_eq!(slices[0].length, 50);

    assert_eq!(slices[1].batch_index, 0);
    assert_eq!(slices[1].token_start_index, 50);
    assert_eq!(slices[1].length, 1);

    assert_eq!(slices[2].batch_index, 1);
    assert_eq!(slices[2].token_start_index, 51);
    assert_eq!(slices[2].length, 1);

    let total: usize = slices.iter().map(|s| s.length).sum();
    assert_eq!(total, 50 + 2);
}

#[test]
fn test_prefill_slices_sum_matches() {
    let slot_list = make_slot_list(vec![make_prefill_state(0, 60), make_prefill_state(100, 80)]);
    let scheduler = Scheduler::new(8, 200, 2, slot_list);

    assert!(scheduler.schedule_batch());
    let task = scheduler.with_task(|t| t.clone());

    assert_eq!(task.prefill_size, 140);

    // prefill token 全部在 slices 前半段（token_start_index < prefill_size）
    let total: usize = task
        .slices
        .iter()
        .filter(|s| s.token_start_index < task.prefill_size)
        .map(|s| s.length.min(task.prefill_size.saturating_sub(s.token_start_index)))
        .sum();
    assert_eq!(total, 140);
}

#[test]
fn test_slot_reuse_workflow() {
    let slot_list = make_slot_list(vec![make_decode_state(0, 0)]);
    let scheduler = Scheduler::new(4, 256, 2, Arc::clone(&slot_list));

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 1);
    });

    slot_list.with_mut(|bl| {
        bl[0].phase = Phase::Eos;
    });
    assert!(!scheduler.schedule_batch());

    slot_list.with_mut(|bl| {
        bl[0].reset_to_start();
        bl[0].start_prefill(100, 50);
    });

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert!(task.prefill_size > 0);
    });
}

#[test]
fn test_work_flag_tracks_task_state() {
    let slot_list = make_slot_list(Vec::new());
    let scheduler = Scheduler::new(8, 512, 1, slot_list);

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
