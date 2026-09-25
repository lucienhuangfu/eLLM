use std::sync::Arc;
use std::time::Duration;

use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{Phase, SessionHandle, SessionMode, SlotState};

use super::test_utils::*;

#[tokio::test]
async fn test_acquire_release_reuse_reusable() {
    let (manager, _buffer) = create_test_manager(4, 5000);

    let h1 = manager.acquire_session("user_a").await.unwrap();
    let slot1 = h1.slot_index;

    let sequence_length = manager
        .batch_states
        .with(|slots| slots[slot1].next_sequence_index);
    Arc::clone(&manager)
        .release_session("user_a", sequence_length)
        .await;

    let h2 = manager.acquire_session("user_a").await.unwrap();
    assert_eq!(h2.slot_index, slot1);
}

#[tokio::test]
async fn test_acquire_release_non_reusable() {
    let (manager, _buffer) = create_test_manager_with_mode(4, 5000, SessionMode::NonReusable);

    let h1 = manager.acquire_session("user_a").await.unwrap();
    let slot1 = h1.slot_index;

    Arc::clone(&manager).release_session("user_a", 10).await;

    let _h2 = manager.acquire_session("user_a").await.unwrap();
}

#[tokio::test]
async fn test_non_reusable_releases_immediately() {
    let (manager, _buffer) = create_test_manager_with_mode(2, 5000, SessionMode::NonReusable);

    let h1 = manager.acquire_session("s1").await.unwrap();
    let h2 = manager.acquire_session("s2").await.unwrap();
    assert_ne!(h1.slot_index, h2.slot_index);

    Arc::clone(&manager).release_session("s1", 10).await;

    let h3 = manager.acquire_session("s3").await.unwrap();
    assert_eq!(h3.slot_index, h1.slot_index);
}

#[tokio::test]
async fn test_slot_reclaimed_after_timeout() {
    let (manager, _buffer) = create_test_manager(4, 100);

    let h1 = manager.acquire_session("timeout_user").await.unwrap();
    let slot1 = h1.slot_index;
    Arc::clone(&manager)
        .release_session("timeout_user", 10)
        .await;

    tokio::time::sleep(Duration::from_millis(200)).await;

    let h2 = manager.acquire_session("other_user").await.unwrap();
    let h3 = manager.acquire_session("another_user").await.unwrap();
    assert_ne!(h2.slot_index, h3.slot_index);

    let _h4 = manager.acquire_session("timeout_user").await.unwrap();
}

#[tokio::test]
async fn test_slot_not_reclaimed_before_timeout() {
    let (manager, _buffer) = create_test_manager(2, 2000);

    let h1 = manager.acquire_session("quick_reuse").await.unwrap();
    let slot1 = h1.slot_index;
    Arc::clone(&manager)
        .release_session("quick_reuse", 10)
        .await;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let h2 = manager.acquire_session("quick_reuse").await.unwrap();
    assert_eq!(h2.slot_index, slot1);
}

#[tokio::test]
async fn test_resumed_session_cancels_timeout() {
    let (manager, _buffer) = create_test_manager(2, 100);

    let h1 = manager.acquire_session("cancel_test").await.unwrap();
    let slot1 = h1.slot_index;

    manager.batch_states.with_mut(|slots| {
        slots[slot1].start_decode(10, 10);
    });

    Arc::clone(&manager)
        .release_session("cancel_test", 10)
        .await;

    let h2 = manager.acquire_session("cancel_test").await.unwrap();
    assert_eq!(h2.slot_index, slot1);

    manager.batch_states.with_mut(|slots| {
        slots[slot1].phase = Phase::Decode;
    });

    tokio::time::sleep(Duration::from_millis(200)).await;

    let phase = manager.batch_states.with(|slots| slots[slot1].phase);
    assert_eq!(phase, Phase::Decode);
}

#[tokio::test]
async fn test_release_nonexistent_session_no_panic() {
    let (manager, _buffer) = create_test_manager(4, 1000);
    Arc::clone(&manager)
        .release_session("does_not_exist", 0)
        .await;
}

#[tokio::test]
async fn test_acquire_same_active_session_returns_reused() {
    let (manager, _buffer) = create_test_manager(4, 1000);

    let h1 = manager.acquire_session("same_session").await.unwrap();
    let h2 = manager.acquire_session("same_session").await.unwrap();

    assert_eq!(h1.slot_index, h2.slot_index);
}

#[tokio::test]
async fn test_multiple_users_concurrent_acquire() {
    let (manager, _buffer) = create_test_manager(8, 5000);

    let user_ids: Vec<String> = (0..5).map(|i| format!("user_{}", i)).collect();
    let mut handles = Vec::new();

    for uid in &user_ids {
        let h = manager.acquire_session(uid).await.unwrap();
        handles.push(h);
    }

    let slot_indices: Vec<usize> = handles.iter().map(|h| h.slot_index).collect();
    let mut unique = slot_indices.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 5);
}

#[tokio::test]
async fn test_user_arrives_and_departs_dynamically() {
    let (manager, _buffer) = create_test_manager(4, 5000);

    let h1 = manager.acquire_session("alice").await.unwrap();
    let h2 = manager.acquire_session("bob").await.unwrap();
    assert_ne!(h1.slot_index, h2.slot_index);

    Arc::clone(&manager).release_session("alice", 50).await;

    let h3 = manager.acquire_session("charlie").await.unwrap();

    let h_alice2 = manager.acquire_session("alice").await.unwrap();
    assert_eq!(h_alice2.slot_index, h1.slot_index);
}

#[tokio::test]
async fn test_session_handle_constructors() {
    let h1 = SessionHandle::new("test".to_string(), 5);
    assert_eq!(h1.session_id, "test");
    assert_eq!(h1.slot_index, 5);
}

#[tokio::test]
async fn test_with_slots_access_patterns() {
    let (manager, _buffer) = create_test_manager(4, 1000);

    let all_start = manager
        .batch_states
        .with(|slots| slots.iter().all(|s| s.phase == Phase::Start));
    assert!(all_start);

    manager.batch_states.with_mut(|slots| {
        slots[0].start_prefill(0, 10);
        slots[1].start_decode(5, 5);
    });

    let phases = manager.batch_states.with(|slots| {
        vec![
            slots[0].phase,
            slots[1].phase,
            slots[2].phase,
            slots[3].phase,
        ]
    });
    assert_eq!(phases[0], Phase::Prefill);
    assert_eq!(phases[1], Phase::Decode);
    assert_eq!(phases[2], Phase::Start);
    assert_eq!(phases[3], Phase::Start);
}

#[tokio::test]
async fn test_full_session_lifecycle_with_scheduler() {
    let batch_size = 4;
    let (manager, _buffer) = create_test_manager(batch_size, 5000);
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        256,
        2,
        Arc::clone(&manager.batch_states),
    ));

    let session_id = "full_lifecycle";

    let handle = manager.acquire_session(session_id).await.unwrap();
    let slot_idx = handle.slot_index;

    let phase = manager.batch_states.with(|slots| slots[slot_idx].phase);
    assert_eq!(phase, Phase::Start);

    run_prefill_and_decode(&manager, &scheduler, slot_idx, 64, 10);

    let sequence_length = manager
        .batch_states
        .with(|slots| slots[slot_idx].next_sequence_index);
    Arc::clone(&manager)
        .release_session(session_id, sequence_length)
        .await;

    let handle2 = manager.acquire_session(session_id).await.unwrap();
    assert_eq!(handle2.slot_index, slot_idx);

    run_prefill_and_decode(&manager, &scheduler, slot_idx, 32, 5);

    let sequence_length2 = manager
        .batch_states
        .with(|slots| slots[slot_idx].next_sequence_index);
    Arc::clone(&manager)
        .release_session(session_id, sequence_length2)
        .await;
}

#[tokio::test]
async fn test_acquire_session_returns_error_when_all_slots_occupied() {
    let batch_size = 2;
    let (manager, _buffer) = create_test_manager_with_mode(batch_size, 5000, SessionMode::NonReusable);

    let h1 = manager.acquire_session("user_1").await.unwrap();
    let h2 = manager.acquire_session("user_2").await.unwrap();
    assert_ne!(h1.slot_index, h2.slot_index);

    let result = manager.acquire_session("user_3").await;
    assert!(result.is_err());

    let err = result.unwrap_err();
    match err {
        crate::serving::ApiError::SlotUnavailable(msg) => {
            assert!(msg.contains("all slots are occupied"));
        }
        _ => panic!("expected SlotUnavailable error"),
    }
}

#[tokio::test]
async fn test_slot_becomes_available_after_release_non_reusable() {
    let batch_size = 2;
    let (manager, _buffer) = create_test_manager_with_mode(batch_size, 5000, SessionMode::NonReusable);

    let h1 = manager.acquire_session("user_1").await.unwrap();
    let _h2 = manager.acquire_session("user_2").await.unwrap();

    assert!(manager.acquire_session("user_3").await.is_err());

    Arc::clone(&manager).release_session("user_1", 10).await;

    let h3 = manager.acquire_session("user_3").await.unwrap();
    assert_eq!(h3.slot_index, h1.slot_index);
}

#[tokio::test]
async fn test_reusable_mode_slot_eviction_when_full() {
    let batch_size = 3;
    let (manager, _buffer) = create_test_manager(batch_size, 100);

    let h1 = manager.acquire_session("user_a").await.unwrap();
    let h2 = manager.acquire_session("user_b").await.unwrap();
    let h3 = manager.acquire_session("user_c").await.unwrap();
    assert_ne!(h1.slot_index, h2.slot_index);
    assert_ne!(h2.slot_index, h3.slot_index);

    Arc::clone(&manager).release_session("user_a", 10).await;
    Arc::clone(&manager).release_session("user_b", 20).await;

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let h4 = manager.acquire_session("user_d").await.unwrap();
    let h5 = manager.acquire_session("user_e").await.unwrap();
    assert_ne!(h4.slot_index, h5.slot_index);

    let h6 = manager.acquire_session("user_f").await;
    assert!(h6.is_err());
    match h6.unwrap_err() {
        crate::serving::ApiError::SlotUnavailable(_) => {}
        _ => panic!("expected SlotUnavailable"),
    }
}
