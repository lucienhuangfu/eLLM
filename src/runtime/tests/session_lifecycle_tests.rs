use std::sync::Arc;
use std::time::Duration;

use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{Phase, SessionHandle, SessionMode, SlotState};

use super::test_utils::*;

#[tokio::test]
async fn test_acquire_release_reuse_reusable() {
    let (manager, _buffer) = create_test_manager(4, 5000);

    let h1 = manager.acquire_session("user_a").await;
    assert!(!h1.is_reused);
    let slot1 = h1.slot_index;

    let token_count = manager
        .batch_states
        .with(|slots| slots[slot1].sequence_index);
    Arc::clone(&manager)
        .release_session("user_a", token_count)
        .await;

    let h2 = manager.acquire_session("user_a").await;
    assert!(h2.is_reused);
    assert_eq!(h2.slot_index, slot1);
}

#[tokio::test]
async fn test_acquire_release_non_reusable() {
    let (manager, _buffer) = create_test_manager_with_mode(4, 5000, SessionMode::NonReusable);

    let h1 = manager.acquire_session("user_a").await;
    assert!(!h1.is_reused);
    let slot1 = h1.slot_index;

    Arc::clone(&manager).release_session("user_a", 10).await;

    let h2 = manager.acquire_session("user_a").await;
    assert!(!h2.is_reused);
}

#[tokio::test]
async fn test_non_reusable_releases_immediately() {
    let (manager, _buffer) = create_test_manager_with_mode(2, 5000, SessionMode::NonReusable);

    let h1 = manager.acquire_session("s1").await;
    let h2 = manager.acquire_session("s2").await;
    assert_ne!(h1.slot_index, h2.slot_index);

    Arc::clone(&manager).release_session("s1", 10).await;

    let h3 = manager.acquire_session("s3").await;
    assert!(!h3.is_reused);
    assert_eq!(h3.slot_index, h1.slot_index);
}

#[tokio::test]
async fn test_slot_reclaimed_after_timeout() {
    let (manager, _buffer) = create_test_manager(4, 100);

    let h1 = manager.acquire_session("timeout_user").await;
    let slot1 = h1.slot_index;
    Arc::clone(&manager)
        .release_session("timeout_user", 10)
        .await;

    tokio::time::sleep(Duration::from_millis(200)).await;

    let h2 = manager.acquire_session("other_user").await;
    let h3 = manager.acquire_session("another_user").await;
    assert!(!h2.is_reused);
    assert!(!h3.is_reused);
    assert_ne!(h2.slot_index, h3.slot_index);

    let h4 = manager.acquire_session("timeout_user").await;
    assert!(!h4.is_reused);
}

#[tokio::test]
async fn test_slot_not_reclaimed_before_timeout() {
    let (manager, _buffer) = create_test_manager(2, 2000);

    let h1 = manager.acquire_session("quick_reuse").await;
    let slot1 = h1.slot_index;
    Arc::clone(&manager)
        .release_session("quick_reuse", 10)
        .await;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let h2 = manager.acquire_session("quick_reuse").await;
    assert!(h2.is_reused);
    assert_eq!(h2.slot_index, slot1);
}

#[tokio::test]
async fn test_resumed_session_cancels_timeout() {
    let (manager, _buffer) = create_test_manager(2, 100);

    let h1 = manager.acquire_session("cancel_test").await;
    let slot1 = h1.slot_index;

    manager.batch_states.with_mut(|slots| {
        slots[slot1].start_decode(10, 10);
    });

    Arc::clone(&manager)
        .release_session("cancel_test", 10)
        .await;

    let h2 = manager.acquire_session("cancel_test").await;
    assert!(h2.is_reused);
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

    let h1 = manager.acquire_session("same_session").await;
    let h2 = manager.acquire_session("same_session").await;

    assert_eq!(h1.slot_index, h2.slot_index);
    assert!(!h1.is_reused);
    assert!(h2.is_reused);
}

#[tokio::test]
async fn test_multiple_users_concurrent_acquire() {
    let (manager, _buffer) = create_test_manager(8, 5000);

    let user_ids: Vec<String> = (0..5).map(|i| format!("user_{}", i)).collect();
    let mut handles = Vec::new();

    for uid in &user_ids {
        let h = manager.acquire_session(uid).await;
        assert!(!h.is_reused);
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

    let h1 = manager.acquire_session("alice").await;
    let h2 = manager.acquire_session("bob").await;
    assert_ne!(h1.slot_index, h2.slot_index);

    Arc::clone(&manager).release_session("alice", 50).await;

    let h3 = manager.acquire_session("charlie").await;
    assert!(!h3.is_reused);

    let h_alice2 = manager.acquire_session("alice").await;
    assert!(h_alice2.is_reused);
    assert_eq!(h_alice2.slot_index, h1.slot_index);
}

#[tokio::test]
async fn test_session_handle_constructors() {
    let h1 = SessionHandle::new("test".to_string(), 5, false);
    assert_eq!(h1.session_id, "test");
    assert_eq!(h1.slot_index, 5);
    assert!(!h1.is_reused);

    let h2 = SessionHandle::new("test".to_string(), 5, true);
    assert_eq!(h2.session_id, "test");
    assert_eq!(h2.slot_index, 5);
    assert!(h2.is_reused);
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

    let handle = manager.acquire_session(session_id).await;
    assert!(!handle.is_reused);
    let slot_idx = handle.slot_index;

    let phase = manager.batch_states.with(|slots| slots[slot_idx].phase);
    assert_eq!(phase, Phase::Start);

    run_prefill_and_decode(&manager, &scheduler, slot_idx, 64, 10);

    let token_count = manager
        .batch_states
        .with(|slots| slots[slot_idx].sequence_index);
    Arc::clone(&manager)
        .release_session(session_id, token_count)
        .await;

    let handle2 = manager.acquire_session(session_id).await;
    assert!(handle2.is_reused);
    assert_eq!(handle2.slot_index, slot_idx);

    run_prefill_and_decode(&manager, &scheduler, slot_idx, 32, 5);

    let token_count2 = manager
        .batch_states
        .with(|slots| slots[slot_idx].sequence_index);
    Arc::clone(&manager)
        .release_session(session_id, token_count2)
        .await;
}
