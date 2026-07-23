use std::sync::Arc;

use crate::num_traits::FromNumber;
use crate::runtime::scheduler::{total_sequence_length, walk_global_range};
use crate::runtime::session::BatchSequence;
use crate::runtime::session::{Phase, SlotState};

use super::test_utils::*;

#[test]
fn test_write_and_read_tokens() {
    let seq_len = 128;
    let batch_size = 2;
    let mut buffer = vec![0usize; batch_size * seq_len];

    let mut batch = BatchSequence::<f32> {
        sequences: buffer.as_mut_ptr(),
        batch_temperature: vec![0.0; batch_size],
        row_size: batch_size,
        col_size: seq_len,
        tokenizer: test_tokenizer(),
        chat_template: test_chat_template(),
    };

    let tokens: Vec<u32> = vec![10, 20, 30, 40, 50];
    let written = batch.write_tokens_at(0, 0, &tokens, 0.7).unwrap();
    assert_eq!(written, 5);
    assert_eq!(batch.batch_temperature[0], 0.7);

    let read_back = batch.token_ids(0, 0, 5);
    assert_eq!(read_back, tokens);
}

#[test]
fn test_write_tokens_at_offset() {
    let seq_len = 128;
    let mut buffer = vec![0usize; seq_len];

    let mut batch = BatchSequence::<f32> {
        sequences: buffer.as_mut_ptr(),
        batch_temperature: vec![0.0; 1],
        row_size: 1,
        col_size: seq_len,
        tokenizer: test_tokenizer(),
        chat_template: test_chat_template(),
    };

    let prefix: Vec<u32> = vec![1, 2, 3, 4, 5];
    batch.write_tokens_at(0, 0, &prefix, 1.0).unwrap();

    let suffix: Vec<u32> = vec![6, 7, 8];
    let written = batch.write_tokens_at(0, 5, &suffix, 1.0).unwrap();
    assert_eq!(written, 3);

    let all = batch.token_ids(0, 0, 8);
    assert_eq!(all, vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_write_tokens_respects_col_size_limit() {
    let seq_len = 8;
    let mut buffer = vec![0usize; seq_len];

    let mut batch = BatchSequence::<f32> {
        sequences: buffer.as_mut_ptr(),
        batch_temperature: vec![0.0; 1],
        row_size: 1,
        col_size: seq_len,
        tokenizer: test_tokenizer(),
        chat_template: test_chat_template(),
    };

    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let written = batch.write_tokens_at(0, 0, &tokens, 1.0).unwrap();
    assert_eq!(written, 8);
}

#[test]
fn test_token_ids_out_of_bounds_returns_empty() {
    let seq_len = 16;
    let mut buffer = vec![0usize; seq_len];

    let batch = BatchSequence::<f32> {
        sequences: buffer.as_mut_ptr(),
        batch_temperature: vec![0.0; 1],
        row_size: 1,
        col_size: seq_len,
        tokenizer: test_tokenizer(),
        chat_template: test_chat_template(),
    };

    assert!(batch.token_ids(0, 0, 0).is_empty());
    assert!(batch.token_ids(0, 20, 30).is_empty());
    assert!(batch.token_ids(0, 10, 5).is_empty());
}

#[tokio::test]
async fn test_get_prefix_match_len_partial_prefix_match() {
    let (manager, _buffer) = create_test_manager(4, 5000);
    let session_id = "partial_test";

    let handle = manager.acquire_session(session_id).await;
    let slot_idx = handle.slot_index;

    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    manager.batch_sequences.with_mut(|bs| {
        bs.write_tokens_at(slot_idx, 0, &tokens, 1.0).unwrap();
    });

    Arc::clone(&manager)
        .release_session(session_id, tokens.len())
        .await;

    manager.acquire_session(session_id).await;

    let new_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 11, 12, 13];
    let prefix_len = manager.prefix_match_len(session_id, &new_tokens).await;
    assert!(prefix_len.is_some());
    assert_eq!(prefix_len.unwrap(), 5);
}

#[tokio::test]
async fn test_get_prefix_match_len_no_prefix_match() {
    let (manager, _buffer) = create_test_manager(4, 5000);
    let session_id = "no_match_test";

    let handle = manager.acquire_session(session_id).await;
    let slot_idx = handle.slot_index;

    let tokens: Vec<u32> = vec![100, 200, 300];
    manager.batch_sequences.with_mut(|bs| {
        bs.write_tokens_at(slot_idx, 0, &tokens, 1.0).unwrap();
    });

    Arc::clone(&manager)
        .release_session(session_id, tokens.len())
        .await;
    manager.acquire_session(session_id).await;

    let new_tokens: Vec<u32> = vec![999, 888];
    let delta = manager.prefix_match_len(session_id, &new_tokens).await;
    assert!(delta.is_none());
}

#[tokio::test]
async fn test_get_prefix_match_len_new_tokens_shorter() {
    let (manager, _buffer) = create_test_manager(4, 5000);
    let session_id = "shorter_test";

    let handle = manager.acquire_session(session_id).await;
    let slot_idx = handle.slot_index;

    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    manager.batch_sequences.with_mut(|bs| {
        bs.write_tokens_at(slot_idx, 0, &tokens, 1.0).unwrap();
    });

    Arc::clone(&manager)
        .release_session(session_id, tokens.len())
        .await;
    manager.acquire_session(session_id).await;

    let new_tokens: Vec<u32> = vec![1, 2, 3];
    let prefix_len = manager.prefix_match_len(session_id, &new_tokens).await;
    assert!(prefix_len.is_some());
    assert_eq!(prefix_len.unwrap(), 3);
}

#[tokio::test]
async fn test_get_prefix_match_len_exact_match() {
    let (manager, _buffer) = create_test_manager(4, 5000);
    let session_id = "exact_test";

    let handle = manager.acquire_session(session_id).await;
    let slot_idx = handle.slot_index;

    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    manager.batch_sequences.with_mut(|bs| {
        bs.write_tokens_at(slot_idx, 0, &tokens, 1.0).unwrap();
    });

    Arc::clone(&manager)
        .release_session(session_id, tokens.len())
        .await;
    manager.acquire_session(session_id).await;

    let prefix_len = manager.prefix_match_len(session_id, &tokens).await;
    assert!(prefix_len.is_some());
    assert_eq!(prefix_len.unwrap(), 5);
}

#[tokio::test]
async fn test_get_prefix_match_len_zero_sequence_length() {
    let (manager, _buffer) = create_test_manager(4, 1000);

    manager.acquire_session("zero_token").await;
    Arc::clone(&manager).release_session("zero_token", 0).await;

    manager.acquire_session("zero_token").await;

    let delta = manager.prefix_match_len("zero_token", &[1, 2, 3]).await;
    assert!(delta.is_none());
}

#[tokio::test]
async fn test_get_prefix_match_len_session_not_found() {
    let (manager, _buffer) = create_test_manager(4, 1000);
    let delta = manager.prefix_match_len("nonexistent", &[1, 2, 3]).await;
    assert!(delta.is_none());
}

#[test]
fn test_total_sequence_length_and_walk_global_range() {
    use crate::runtime::scheduler::SequenceSlice;

    let slices = vec![
        SequenceSlice {
            batch_index: 0,
            next_sequence_index: 0,
            token_start_index: 0,
            length: 6,
            last_token_flag: false,
        },
        SequenceSlice {
            batch_index: 1,
            next_sequence_index: 0,
            token_start_index: 6,
            length: 2,
            last_token_flag: false,
        },
    ];

    assert_eq!(total_sequence_length(&slices), 8);

    let mut visited = Vec::new();
    walk_global_range(&slices, 4, 8, |g, b, s| visited.push((g, b, s)));
    assert_eq!(visited, vec![(4, 0, 4), (5, 0, 5), (6, 1, 0), (7, 1, 1)]);
}

#[test]
fn test_slot_state_phase_transitions() {
    let mut state = SlotState::idle();
    assert_eq!(state.phase, Phase::Start);
    assert!(state.is_available());

    state.start_prefill(0, 100);
    assert_eq!(state.phase, Phase::Prefill);
    assert!(!state.is_available());
    assert_eq!(state.filling_length(), 100);

    advance_slot(&mut state, 50);
    assert_eq!(state.phase, Phase::Prefill);
    assert_eq!(state.filling_length(), 50);

    advance_slot(&mut state, 50);
    assert_eq!(state.phase, Phase::Decode);
    assert!(!state.is_available());

    advance_slot(&mut state, 20);
    assert_eq!(state.phase, Phase::Decode);
    assert_eq!(state.next_sequence_index, 120);

    state.phase = Phase::Eos;
    assert_eq!(state.phase, Phase::Eos);
    assert!(state.is_available());

    state.reset_to_start();
    assert_eq!(state.phase, Phase::Start);
    assert!(state.is_available());
    assert_eq!(state.next_sequence_index, usize::MAX);
    assert_eq!(state.filling_length(), 0);
}
