use std::sync::Arc;
use std::time::Duration;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch::BatchSequence;
use crate::runtime::scheduler::{BatchMode, Scheduler};
use crate::runtime::session::{Phase, SessionMode, SessionHandle, SlotManager, SlotState};

fn model_dir() -> String {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("models");
    p.push("MiniMax-M2.5");
    p.to_string_lossy().into_owned()
}

fn create_test_manager(batch_size: usize, timeout_ms: u64) -> Arc<SlotManager<f16>> {
    let dir = model_dir();
    let batch_sequences = Arc::new(SharedMut::new(
        BatchSequence::<f16>::new(
            std::ptr::null_mut(),
            batch_size,
            1024,
            &format!("{}/tokenizer.json", dir),
            &format!("{}/tokenizer_config.json", dir),
            &format!("{}/chat_template.jinja", dir),
        )
        .unwrap(),
    ));
    let batch_states = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect::<Vec<_>>(),
    ));
    Arc::new(SlotManager::new(
        batch_size,
        batch_sequences,
        batch_states,
        SessionMode::Reusable,
        timeout_ms,
    ))
}

fn create_manager_with_mode(
    batch_size: usize,
    timeout_ms: u64,
    mode: SessionMode,
) -> Arc<SlotManager<f16>> {
    let dir = model_dir();
    let batch_sequences = Arc::new(SharedMut::new(
        BatchSequence::<f16>::new(
            std::ptr::null_mut(),
            batch_size,
            1024,
            &format!("{}/tokenizer.json", dir),
            &format!("{}/tokenizer_config.json", dir),
            &format!("{}/chat_template.jinja", dir),
        )
        .unwrap(),
    ));
    let batch_states = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect::<Vec<_>>(),
    ));
    Arc::new(SlotManager::new(
        batch_size,
        batch_sequences,
        batch_states,
        mode,
        timeout_ms,
    ))
}

fn advance_slot(slot: &mut SlotState, steps: usize) {
    slot.sequence_index += steps;
    if slot.phase == Phase::Prefill {
        slot.filling_length = slot.filling_length.saturating_sub(steps);
        if slot.filling_length == 0 {
            slot.phase = Phase::Decode;
        }
    }
}

fn run_prefill_and_decode(
    manager: &SlotManager<f16>,
    scheduler: &Scheduler,
    slot_index: usize,
    prefill_len: usize,
    decode_steps: usize,
) {
    manager.with_slots_mut(|slots| {
        slots[slot_index] = SlotState::new_prefill_state(0, prefill_len);
    });

    assert!(scheduler.schedule_batch());

    manager.with_slots_mut(|slots| {
        advance_slot(&mut slots[slot_index], prefill_len);
    });

    for _ in 0..decode_steps {
        assert!(scheduler.schedule_batch());
        manager.with_slots_mut(|slots| {
            advance_slot(&mut slots[slot_index], 1);
        });
    }

    manager.with_slots_mut(|slots| {
        slots[slot_index].phase = Phase::Eos;
    });
}

// ── 1. Complete User Session Lifecycle ─────────────────────

#[tokio::test]
async fn test_full_user_session_lifecycle_reusable() {
    let batch_size = 4;
    let manager = create_test_manager(batch_size, 5000);
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        256,
        2,
        manager.batch_list(),
    ));

    let session_id = "user_session_001";

    let handle = manager.acquire_session(session_id).await.unwrap();
    assert!(!handle.is_reused);
    assert_eq!(handle.session_id, session_id);
    let slot_idx = handle.slot_index;

    let phase = manager.with_slots(|slots| slots[slot_idx].phase);
    assert_eq!(phase, Phase::Start);

    run_prefill_and_decode(&manager, &scheduler, slot_idx, 64, 10);

    let token_count = manager.with_slots(|slots| slots[slot_idx].sequence_index);
    manager.release_session(session_id, token_count).await;

    let handle2 = manager.acquire_session(session_id).await.unwrap();
    assert!(handle2.is_reused);
    assert_eq!(handle2.slot_index, slot_idx);
    assert_eq!(handle2.session_id, session_id);

    run_prefill_and_decode(&manager, &scheduler, slot_idx, 32, 5);

    let token_count2 = manager.with_slots(|slots| slots[slot_idx].sequence_index);
    manager.release_session(session_id, token_count2).await;
}

#[tokio::test]
async fn test_full_user_session_lifecycle_non_reusable() {
    let batch_size = 4;
    let manager = create_manager_with_mode(batch_size, 5000, SessionMode::NonReusable);

    let session_id = "user_session_001";

    let handle1 = manager.acquire_session(session_id).await.unwrap();
    assert!(!handle1.is_reused);
    let slot1 = handle1.slot_index;

    manager.release_session(session_id, 100).await;

    let handle2 = manager.acquire_session(session_id).await.unwrap();
    assert!(!handle2.is_reused);
}

// ── 2. Multi-User Concurrent Session Tests ─────────────────

#[tokio::test]
async fn test_multiple_users_concurrent_sessions() {
    let batch_size = 8;
    let manager = create_test_manager(batch_size, 5000);
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        512,
        4,
        manager.batch_list(),
    ));

    let user_ids: Vec<String> = (0..5).map(|i| format!("user_{}", i)).collect();

    let mut handles = Vec::new();
    for uid in &user_ids {
        let h = manager.acquire_session(uid).await.unwrap();
        assert!(!h.is_reused);
        handles.push(h);
    }

    let slot_indices: Vec<usize> = handles.iter().map(|h| h.slot_index).collect();
    let mut unique = slot_indices.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 5);

    for &slot_idx in &slot_indices {
        manager.with_slots_mut(|slots| {
            slots[slot_idx] = SlotState::new_prefill_state(0, 32);
        });
    }

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.mode, BatchMode::Prefill);
        assert!(task.prefill_size > 0);
    });

    for &slot_idx in &slot_indices {
        manager.with_slots_mut(|slots| {
            let fl = slots[slot_idx].filling_length;
            advance_slot(&mut slots[slot_idx], fl);
        });
    }

    for step in 0..5 {
        assert!(scheduler.schedule_batch());

        let expected_decode = if step < 3 { 5 } else { 3 };
        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, expected_decode);
        });

        for &slot_idx in &slot_indices {
            manager.with_slots_mut(|slots| {
                if slots[slot_idx].phase == Phase::Decode {
                    advance_slot(&mut slots[slot_idx], 1);
                }
            });
        }

        if step == 2 {
            manager.with_slots_mut(|slots| {
                slots[slot_indices[0]].phase = Phase::Eos;
                slots[slot_indices[2]].phase = Phase::Eos;
            });
        }
    }

    let active_count = manager.with_slots(|slots| {
        slots
            .iter()
            .filter(|s| s.phase == Phase::Decode)
            .count()
    });
    assert_eq!(active_count, 3);

    for (i, uid) in user_ids.iter().enumerate() {
        let token_count = manager
            .with_slots(|slots| slots[slot_indices[i]].sequence_index);
        manager.release_session(uid, token_count).await;
    }
}

#[tokio::test]
async fn test_user_arrives_and_departs_dynamically() {
    let batch_size = 4;
    let manager = create_test_manager(batch_size, 5000);

    let h1 = manager.acquire_session("alice").await.unwrap();
    let h2 = manager.acquire_session("bob").await.unwrap();
    assert_ne!(h1.slot_index, h2.slot_index);

    manager.release_session("alice", 50).await;

    let h3 = manager.acquire_session("charlie").await.unwrap();
    assert!(!h3.is_reused);

    let h_alice2 = manager.acquire_session("alice").await.unwrap();
    assert!(h_alice2.is_reused);
    assert_eq!(h_alice2.slot_index, h1.slot_index);
}

// ── 3. Slot Reuse and Eviction Tests ───────────────────────

#[tokio::test]
async fn test_slot_reused_from_reserved_before_timeout() {
    let batch_size = 4;
    let manager = create_test_manager(batch_size, 10000);

    let h1 = manager.acquire_session("user_a").await.unwrap();
    assert!(!h1.is_reused);
    let slot1 = h1.slot_index;

    manager.release_session("user_a", 10).await;

    let h2 = manager.acquire_session("user_a").await.unwrap();
    assert!(h2.is_reused);
    assert_eq!(h2.slot_index, slot1);
}

#[tokio::test]
async fn test_reserved_slot_takes_priority_over_lru() {
    let batch_size = 3;
    let manager = create_test_manager(batch_size, 10000);

    let h1 = manager.acquire_session("user_a").await.unwrap();
    let h2 = manager.acquire_session("user_b").await.unwrap();
    assert_ne!(h1.slot_index, h2.slot_index);

    manager.release_session("user_a", 10).await;

    let h_a2 = manager.acquire_session("user_a").await.unwrap();
    assert!(h_a2.is_reused);
    assert_eq!(h_a2.slot_index, h1.slot_index);

    let h3 = manager.acquire_session("user_c").await.unwrap();
    assert!(!h3.is_reused);
}

#[tokio::test]
async fn test_all_slots_active_no_new_available() {
    let batch_size = 2;
    let manager = create_test_manager(batch_size, 10000);

    let h1 = manager.acquire_session("user_a").await.unwrap();
    let h2 = manager.acquire_session("user_b").await.unwrap();
    assert_ne!(h1.slot_index, h2.slot_index);

    manager.release_session("user_a", 10).await;
    manager.release_session("user_b", 20).await;

    let h_a2 = manager.acquire_session("user_a").await.unwrap();
    assert!(h_a2.is_reused);
    let h_b2 = manager.acquire_session("user_b").await.unwrap();
    assert!(h_b2.is_reused);

    assert_ne!(h_a2.slot_index, h_b2.slot_index);
}

// ── 4. Timeout-based Reclaim Tests ─────────────────────────

#[tokio::test]
async fn test_slot_reclaimed_after_timeout() {
    let batch_size = 4;
    let manager = create_test_manager(batch_size, 200);

    let h1 = manager.acquire_session("session_timeout").await.unwrap();
    let slot1 = h1.slot_index;
    manager.release_session("session_timeout", 10).await;

    tokio::time::sleep(Duration::from_millis(300)).await;

    let h2 = manager.acquire_session("other_session").await.unwrap();
    assert!(!h2.is_reused);

    let h3 = manager.acquire_session("another_session").await.unwrap();
    assert!(!h3.is_reused);
    assert_ne!(h2.slot_index, h3.slot_index);

    let h4 = manager.acquire_session("session_timeout").await.unwrap();
    assert!(!h4.is_reused);
}

#[tokio::test]
async fn test_slot_not_reclaimed_before_timeout() {
    let batch_size = 2;
    let manager = create_test_manager(batch_size, 2000);

    let h1 = manager.acquire_session("quick_reuse").await.unwrap();
    let slot1 = h1.slot_index;
    manager.release_session("quick_reuse", 10).await;

    tokio::time::sleep(Duration::from_millis(200)).await;

    let h2 = manager.acquire_session("quick_reuse").await.unwrap();
    assert!(h2.is_reused);
    assert_eq!(h2.slot_index, slot1);
}

#[tokio::test]
async fn test_reuse_cancels_timeout_reset() {
    let batch_size = 2;
    let manager = create_test_manager(batch_size, 200);

    let h1 = manager.acquire_session("cancel_test").await.unwrap();
    let slot1 = h1.slot_index;

    manager.with_slots_mut(|slots| {
        slots[slot1] = SlotState::new_decode_state(10, 10);
    });

    manager.release_session("cancel_test", 10).await;

    let h2 = manager.acquire_session("cancel_test").await.unwrap();
    assert!(h2.is_reused);
    assert_eq!(h2.slot_index, slot1);

    manager.with_slots_mut(|slots| {
        slots[slot1].phase = Phase::Decode;
    });

    tokio::time::sleep(Duration::from_millis(300)).await;

    let phase = manager.with_slots(|slots| slots[slot1].phase);
    assert_eq!(phase, Phase::Decode);

    manager.release_session("cancel_test", 20).await;
}

// ── 5. Edge Case and Boundary Tests ────────────────────────

#[tokio::test]
async fn test_release_nonexistent_session_no_panic() {
    let manager = create_test_manager(4, 1000);
    manager.release_session("does_not_exist", 0).await;
}

#[tokio::test]
async fn test_acquire_same_session_active() {
    let manager = create_test_manager(4, 1000);

    let h1 = manager.acquire_session("same_session").await.unwrap();
    let h2 = manager.acquire_session("same_session").await.unwrap();

    assert_eq!(h1.slot_index, h2.slot_index);
    assert!(!h1.is_reused);
    assert!(h2.is_reused);
}

#[tokio::test]
async fn test_slot_state_phase_transitions() {
    let mut state = SlotState::new_start_state();
    assert_eq!(state.phase, Phase::Start);
    assert!(state.is_available());

    state = SlotState::new_prefill_state(0, 100);
    assert_eq!(state.phase, Phase::Prefill);
    assert!(!state.is_available());
    assert_eq!(state.filling_length, 100);

    advance_slot(&mut state, 50);
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length, 50);

    advance_slot(&mut state, 50);
    assert_eq!(state.phase, Phase::Decode);
    assert!(!state.is_available());

    advance_slot(&mut state, 20);
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.sequence_index, 120);

    state.phase = Phase::Eos;
    assert_eq!(state.phase, Phase::Eos);
    assert!(state.is_available());

    state.reset_to_start();
    assert_eq!(state.phase, Phase::Start);
    assert!(state.is_available());
    assert_eq!(state.sequence_index, usize::MAX);
    assert_eq!(state.filling_length, 0);
}

#[tokio::test]
async fn test_session_handle_constructors() {
    let h1 = SessionHandle::new("test".to_string(), 5);
    assert_eq!(h1.session_id, "test");
    assert_eq!(h1.slot_index, 5);
    assert!(!h1.is_reused);

    let h2 = SessionHandle::reused("test".to_string(), 5);
    assert_eq!(h2.session_id, "test");
    assert_eq!(h2.slot_index, 5);
    assert!(h2.is_reused);
}

#[tokio::test]
async fn test_with_slots_and_with_slots_mut() {
    let manager = create_test_manager(4, 1000);

    let all_start = manager.with_slots(|slots| slots.iter().all(|s| s.phase == Phase::Start));
    assert!(all_start);

    manager.with_slots_mut(|slots| {
        slots[0] = SlotState::new_prefill_state(0, 10);
        slots[1] = SlotState::new_decode_state(5, 5);
    });

    let phases = manager.with_slots(|slots| {
        vec![slots[0].phase, slots[1].phase, slots[2].phase, slots[3].phase]
    });
    assert_eq!(phases[0], Phase::Prefill);
    assert_eq!(phases[1], Phase::Decode);
    assert_eq!(phases[2], Phase::Start);
    assert_eq!(phases[3], Phase::Start);
}

// ── 7. Realistic Multi-Round Chat Simulation ───────────────

#[tokio::test]
async fn test_multi_round_chat_with_slot_reuse() {
    let batch_size = 4;
    let manager = create_test_manager(batch_size, 5000);
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        512,
        2,
        manager.batch_list(),
    ));

    let user_id = "chat_user_42";

    let mut current_slot = 0;
    let mut total_tokens = 0usize;

    for round in 1..=3 {
        let handle = manager.acquire_session(user_id).await.unwrap();
        let slot_idx = handle.slot_index;

        if round > 1 {
            assert!(handle.is_reused, "round {} should reuse slot", round);
            assert_eq!(slot_idx, current_slot);
        } else {
            assert!(!handle.is_reused);
            current_slot = slot_idx;
        }

        let prefill_len = 20 + round * 10;
        let decode_steps = 5 + round * 2;

        manager.with_slots_mut(|slots| {
            slots[slot_idx] = SlotState::new_prefill_state(total_tokens, prefill_len);
        });

        assert!(scheduler.schedule_batch());
        manager.with_slots_mut(|slots| {
            advance_slot(&mut slots[slot_idx], prefill_len);
        });

        for _ in 0..decode_steps {
            assert!(scheduler.schedule_batch());
            manager.with_slots_mut(|slots| {
                advance_slot(&mut slots[slot_idx], 1);
            });
        }

        manager.with_slots_mut(|slots| {
            slots[slot_idx].phase = Phase::Eos;
        });

        total_tokens = manager
            .with_slots(|slots| slots[slot_idx].sequence_index);

        manager.release_session(user_id, total_tokens).await;
    }
}

#[tokio::test]
async fn test_concurrent_multi_user_chat_simulation() {
    let batch_size = 6;
    let manager = create_test_manager(batch_size, 5000);
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        512,
        4,
        manager.batch_list(),
    ));

    let users: Vec<String> = vec![
        "alice".to_string(),
        "bob".to_string(),
        "charlie".to_string(),
    ];

    for round in 0..3 {
        let mut handles = Vec::new();
        for user in &users {
            let h = manager.acquire_session(user).await.unwrap();
            handles.push(h);
        }

        for (i, handle) in handles.iter().enumerate() {
            let prefill_len = 16 + i * 8 + round * 4;
            manager.with_slots_mut(|slots| {
                slots[handle.slot_index] = SlotState::new_prefill_state(0, prefill_len);
            });
        }

        assert!(scheduler.schedule_batch());

        for handle in &handles {
            manager.with_slots_mut(|slots| {
                let fl = slots[handle.slot_index].filling_length;
                advance_slot(&mut slots[handle.slot_index], fl);
            });
        }

        for _ in 0..5 {
            assert!(scheduler.schedule_batch());
            for handle in &handles {
                manager.with_slots_mut(|slots| {
                    if slots[handle.slot_index].phase == Phase::Decode {
                        advance_slot(&mut slots[handle.slot_index], 1);
                    }
                });
            }
        }

        for (i, user) in users.iter().enumerate() {
            manager.with_slots_mut(|slots| {
                slots[handles[i].slot_index].phase = Phase::Eos;
            });
            let tc = manager
                .with_slots(|slots| slots[handles[i].slot_index].sequence_index);
            manager.release_session(user, tc).await;
        }
    }

    for user in &users {
        let h = manager.acquire_session(user).await.unwrap();
        assert!(h.is_reused);
        manager.release_session(user, 100).await;
    }
}

// ── 8. Mixed Workflow with Scheduler Integration ───────────

#[test]
fn test_scheduler_with_slot_state_transitions() {
    let batch_size = 6;
    let batch_list = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect::<Vec<_>>(),
    ));
    let scheduler = Scheduler::new(batch_size, 256, 2, Arc::clone(&batch_list));

    assert!(!scheduler.schedule_batch());

    batch_list.with_mut(|bl| {
        bl[0] = SlotState::new_prefill_state(0, 64);
        bl[1] = SlotState::new_prefill_state(100, 32);
    });
    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.mode, BatchMode::Prefill);
        assert_eq!(task.decode_size, 0);
        assert!(task.prefill_size > 0);
    });

    batch_list.with_mut(|bl| {
        advance_slot(&mut bl[0], 64);
        advance_slot(&mut bl[1], 32);
        bl[2] = SlotState::new_prefill_state(200, 48);
    });
    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.mode, BatchMode::Mixed);
        assert_eq!(task.decode_size, 2);
        assert!(task.prefill_size > 0);
    });

    batch_list.with_mut(|bl| {
        advance_slot(&mut bl[2], 48);
    });
    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.mode, BatchMode::Decode);
        assert_eq!(task.decode_size, 3);
        assert_eq!(task.prefill_size, 0);
    });

    batch_list.with_mut(|bl| {
        bl[0].phase = Phase::Eos;
    });
    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert_eq!(task.decode_size, 2);
    });

    batch_list.with_mut(|bl| {
        bl[1].phase = Phase::Eos;
        bl[2].phase = Phase::Eos;
    });
    assert!(!scheduler.schedule_batch());
}
