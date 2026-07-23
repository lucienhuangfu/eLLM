use std::sync::Arc;

use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{Phase, SessionMode, SlotState};

use super::test_utils::*;

#[tokio::test]
async fn test_multi_round_chat_with_kv_cache_reuse() {
    let batch_size = 4;
    let (manager, _buffer) = create_test_manager(batch_size, 5000);
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        512,
        2,
        Arc::clone(&manager.batch_states),
    ));

    let user_id = "chat_user_42";
    let mut current_slot = 0;
    let mut total_tokens = 0usize;

    for round in 1..=3 {
        let handle = manager.acquire_session(user_id).await;
        let slot_idx = handle.slot_index;

        if round > 1 {
            assert_eq!(slot_idx, current_slot);
        } else {
            current_slot = slot_idx;
        }

        let prefill_length = 20 + round * 10;
        let decode_steps = 5 + round * 2;

        manager.batch_states.with_mut(|slots| {
            slots[slot_idx].start_prefill(total_tokens, prefill_length);
        });

        assert!(scheduler.schedule_batch());
        manager.batch_states.with_mut(|slots| {
            advance_slot(&mut slots[slot_idx], prefill_length);
        });

        for _ in 0..decode_steps {
            assert!(scheduler.schedule_batch());
            manager.batch_states.with_mut(|slots| {
                advance_slot(&mut slots[slot_idx], 1);
            });
        }

        manager.batch_states.with_mut(|slots| {
            slots[slot_idx].phase = Phase::Eos;
        });

        total_tokens = manager
            .batch_states
            .with(|slots| slots[slot_idx].next_sequence_index);

        Arc::clone(&manager)
            .release_session(user_id, total_tokens)
            .await;
    }
}

#[tokio::test]
async fn test_concurrent_multi_user_chat_simulation() {
    let batch_size = 6;
    let (manager, _buffer) = create_test_manager(batch_size, 5000);
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        512,
        4,
        Arc::clone(&manager.batch_states),
    ));

    let users: Vec<String> = vec!["alice".into(), "bob".into(), "charlie".into()];

    for round in 0..3 {
        let mut handles = Vec::new();
        for user in &users {
            let h = manager.acquire_session(user).await;
            handles.push(h);
        }

        for (i, handle) in handles.iter().enumerate() {
            let prefill_length = 16 + i * 8 + round * 4;
            manager.batch_states.with_mut(|slots| {
                slots[handle.slot_index].start_prefill(0, prefill_length);
            });
        }

        assert!(scheduler.schedule_batch());

        for handle in &handles {
            manager.batch_states.with_mut(|slots| {
                let fl = slots[handle.slot_index].filling_length();
                advance_slot(&mut slots[handle.slot_index], fl);
            });
        }

        for _ in 0..5 {
            assert!(scheduler.schedule_batch());
            for handle in &handles {
                manager.batch_states.with_mut(|slots| {
                    if slots[handle.slot_index].phase == Phase::Decode {
                        advance_slot(&mut slots[handle.slot_index], 1);
                    }
                });
            }
        }

        for (i, user) in users.iter().enumerate() {
            manager.batch_states.with_mut(|slots| {
                slots[handles[i].slot_index].phase = Phase::Eos;
            });
            let tc = manager
                .batch_states
                .with(|slots| slots[handles[i].slot_index].next_sequence_index);
            Arc::clone(&manager).release_session(user, tc).await;
        }
    }

    for user in &users {
        let h = manager.acquire_session(user).await;
        Arc::clone(&manager).release_session(user, 100).await;
    }
}

#[tokio::test]
async fn test_incremental_prefill_with_prefix_match() {
    let (manager, _buffer) = create_test_manager(4, 5000);
    let session_id = "incremental_prefill_test";

    let handle = manager.acquire_session(session_id).await;
    let slot_idx = handle.slot_index;

    let round1_tokens: Vec<u32> = (1..=20).collect();
    manager.batch_sequences.with_mut(|bs| {
        bs.write_tokens_at(slot_idx, 0, &round1_tokens, 1.0)
            .unwrap();
    });
    Arc::clone(&manager)
        .release_session(session_id, round1_tokens.len())
        .await;

    let handle2 = manager.acquire_session(session_id).await;
    assert_eq!(handle2.slot_index, slot_idx);

    let round2_tokens: Vec<u32> = (1..=15).chain(100..110).collect();
    let prefix_len = manager.prefix_match_len(session_id, &round2_tokens).await;

    assert!(prefix_len.is_some());
    let prefix = prefix_len.unwrap();
    assert_eq!(prefix, 15);

    let delta_tokens = &round2_tokens[prefix..];
    assert_eq!(delta_tokens.len(), 10);

    manager.batch_sequences.with_mut(|bs| {
        let written = bs
            .write_tokens_at(slot_idx, prefix, delta_tokens, 1.0)
            .unwrap();
        assert_eq!(written, delta_tokens.len());
    });

    let verified = manager
        .batch_sequences
        .with(|bs| bs.token_ids(slot_idx, 0, prefix + delta_tokens.len()));
    assert_eq!(verified, round2_tokens);
}

#[tokio::test]
async fn test_mixed_reusable_and_non_reusable_sessions() {
    let batch_size = 4;
    let (reusable_manager, _buf1) = create_test_manager(batch_size, 5000);
    let (non_reusable_manager, _buf2) =
        create_test_manager_with_mode(batch_size, 5000, SessionMode::NonReusable);

    let h1 = reusable_manager.acquire_session("user_r").await;
    Arc::clone(&reusable_manager)
        .release_session("user_r", 10)
        .await;
    let _h2 = reusable_manager.acquire_session("user_r").await;

    let h3 = non_reusable_manager.acquire_session("user_nr").await;
    Arc::clone(&non_reusable_manager)
        .release_session("user_nr", 10)
        .await;
    let _h4 = non_reusable_manager.acquire_session("user_nr").await;
}

#[tokio::test]
async fn test_session_eviction_when_all_slots_full() {
    let batch_size = 4;
    let (manager, _buffer) = create_test_manager(batch_size, 100);

    let h1 = manager.acquire_session("user_1").await;
    let h2 = manager.acquire_session("user_2").await;
    assert_ne!(h1.slot_index, h2.slot_index);

    Arc::clone(&manager).release_session("user_1", 10).await;
    Arc::clone(&manager).release_session("user_2", 20).await;

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let h3 = manager.acquire_session("user_3").await;
    let h4 = manager.acquire_session("user_4").await;
    assert_ne!(h3.slot_index, h4.slot_index);

    let _h1_again = manager.acquire_session("user_1").await;
}

#[tokio::test]
async fn test_multiple_users_with_mixed_phases_in_scheduler() {
    let batch_size = 8;
    let (manager, _buffer) = create_test_manager(batch_size, 5000);
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        512,
        4,
        Arc::clone(&manager.batch_states),
    ));

    let user_ids: Vec<String> = (0..5).map(|i| format!("user_{}", i)).collect();
    let mut handles = Vec::new();

    for uid in &user_ids {
        let h = manager.acquire_session(uid).await;
        handles.push(h);
    }

    for (i, handle) in handles.iter().enumerate() {
        manager.batch_states.with_mut(|slots| {
            slots[handle.slot_index].start_prefill(0, 32 + i * 8);
        });
    }

    assert!(scheduler.schedule_batch());
    scheduler.with_task(|task| {
        assert!(task.prefill_size > 0);
    });

    for handle in &handles {
        manager.batch_states.with_mut(|slots| {
            let fl = slots[handle.slot_index].filling_length();
            advance_slot(&mut slots[handle.slot_index], fl);
        });
    }

    for step in 0..5 {
        assert!(scheduler.schedule_batch());

        let expected_decode = if step < 3 { 5 } else { 3 };
        scheduler.with_task(|task| {
            assert_eq!(task.decode_size, expected_decode);
        });

        for handle in &handles {
            manager.batch_states.with_mut(|slots| {
                if slots[handle.slot_index].phase == Phase::Decode {
                    advance_slot(&mut slots[handle.slot_index], 1);
                }
            });
        }

        if step == 2 {
            manager.batch_states.with_mut(|slots| {
                slots[handles[0].slot_index].phase = Phase::Eos;
                slots[handles[2].slot_index].phase = Phase::Eos;
            });
        }
    }

    let active_count = manager
        .batch_states
        .with(|slots| slots.iter().filter(|s| s.phase == Phase::Decode).count());
    assert_eq!(active_count, 3);

    for (i, uid) in user_ids.iter().enumerate() {
        let sequence_length = manager
            .batch_states
            .with(|slots| slots[handles[i].slot_index].next_sequence_index);
        Arc::clone(&manager).release_session(uid, sequence_length).await;
    }
}
