use axum::http::{Request, StatusCode};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::runtime::session::{Phase, SessionMode};
use crate::serving::ChatMessage;

use super::test_utils::*;

const EOS_TEXT_QWEN3: &str = "<|im_end|>";

fn setup_qwen3_fakeecho(
    batch_size: usize,
    mode: SessionMode,
    threads: usize,
    max_gen: usize,
    timeout_ms: u64,
) -> (axum::Router, Arc<crate::runtime::session::SlotManager<f16>>) {
    let (manager, _buffer) =
        create_qwen3_test_manager_with_mode(batch_size, timeout_ms, mode);
    let router = crate::serving::server::build_router(Arc::clone(&manager));
    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let gen_tokens: Vec<usize> = (0..max_gen).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, threads, gen_tokens);
    (router, manager)
}

// ─── Overload protection (sync) ────────────────────────────────────────────

#[tokio::test]
async fn test_overload_sync_returns_503() {
    let (router, manager) = setup_qwen3_fakeecho(2, SessionMode::NonReusable, 1, 10, 5000);

    let _h1 = manager.acquire_session("active_user_1").await.unwrap();
    let _h2 = manager.acquire_session("active_user_2").await.unwrap();

    let body = simple_chat_body("Hello overload", false);
    let request = chat_request(&body);
    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let retry_after = response
        .headers()
        .get(axum::http::header::RETRY_AFTER)
        .expect("Retry-After header should be present");
    assert_eq!(retry_after, "1");

    let resp_body = collect_body(response.into_body()).await;
    let body_str = String::from_utf8_lossy(&resp_body);
    assert!(body_str.contains("Service unavailable"));
    assert!(body_str.contains("all slots are occupied"));
}

#[tokio::test]
async fn test_overload_sync_recovery_after_release() {
    let (router, manager) = setup_qwen3_fakeecho(2, SessionMode::NonReusable, 1, 10, 5000);

    let _h1 = manager.acquire_session("held_user_1").await.unwrap();
    let _h2 = manager.acquire_session("held_user_2").await.unwrap();

    let body1 = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Should fail"}],
        "stream": false,
        "session_id": "new_user_1"
    });

    let request1 = chat_request(&body1);
    let response1 = router.clone().oneshot(request1).await.unwrap();
    assert_eq!(response1.status(), StatusCode::SERVICE_UNAVAILABLE);

    drop(_h1);
    drop(_h2);
    manager.release_session("held_user_1", 0).await;

    let body2 = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Should succeed now"}],
        "stream": false,
        "session_id": "new_user_2"
    });

    let request2 = chat_request(&body2);
    let response2 = router.clone().oneshot(request2).await.unwrap();
    assert_eq!(response2.status(), StatusCode::OK);
}

// ─── Overload protection (stream) ──────────────────────────────────────────

#[tokio::test]
async fn test_overload_stream_returns_503() {
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        2,
        5000,
        SessionMode::NonReusable,
        false,
        false,
    );
    let router = crate::serving::server::build_router(Arc::clone(&manager));
    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let gen_tokens: Vec<usize> = (0..10).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, gen_tokens);

    let _h1 = manager.acquire_session("active_user_1").await.unwrap();
    let _h2 = manager.acquire_session("active_user_2").await.unwrap();

    let body = simple_chat_body("Hello overload", true);
    let request = chat_request(&body);
    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let retry_after = response
        .headers()
        .get(axum::http::header::RETRY_AFTER)
        .expect("Retry-After header should be present");
    assert_eq!(retry_after, "1");

    let resp_body = collect_body(response.into_body()).await;
    let body_str = String::from_utf8_lossy(&resp_body);
    assert!(body_str.contains("Service unavailable"));
    assert!(body_str.contains("all slots are occupied"));
}

#[tokio::test]
async fn test_overload_stream_recovery_after_release() {
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        2,
        5000,
        SessionMode::NonReusable,
        false,
        false,
    );
    let router = crate::serving::server::build_router(Arc::clone(&manager));
    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let gen_tokens: Vec<usize> = (0..10).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, gen_tokens);

    let _h1 = manager.acquire_session("held_user_1").await.unwrap();
    let _h2 = manager.acquire_session("held_user_2").await.unwrap();

    let body1 = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Should fail"}],
        "stream": true,
        "session_id": "new_user_1"
    });

    let request1 = chat_request(&body1);
    let response1 = router.clone().oneshot(request1).await.unwrap();
    assert_eq!(response1.status(), StatusCode::SERVICE_UNAVAILABLE);

    drop(_h1);
    drop(_h2);
    manager.release_session("held_user_1", 0).await;

    let body2 = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Should succeed now"}],
        "stream": true,
        "session_id": "new_user_2"
    });

    let request2 = chat_request(&body2);
    let response2 = router.clone().oneshot(request2).await.unwrap();
    assert_eq!(response2.status(), StatusCode::OK);

    let content_type = response2.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("text/event-stream"));
}

// ─── Reusable session (sync) ───────────────────────────────────────────────

#[tokio::test]
async fn test_reusable_session_two_requests_same_session() {
    let (router, _manager) = setup_qwen3_fakeecho(4, SessionMode::Reusable, 1, 10, 5000);

    let session_id = "reusable-session-1";

    let body1 = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "session_id": session_id
    });

    let request1 = chat_request(&body1);
    let response1 = router.clone().oneshot(request1).await.unwrap();
    assert_eq!(response1.status(), StatusCode::OK);

    let resp_body1 = collect_body(response1.into_body()).await;
    let json1: serde_json::Value = serde_json::from_slice(&resp_body1).unwrap();
    let content1 = json1["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert!(!content1.is_empty());
    assert!(content1.ends_with(EOS_TEXT_QWEN3));

    tokio::time::sleep(Duration::from_millis(50)).await;

    let body2 = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ],
        "stream": false,
        "session_id": session_id
    });

    let request2 = chat_request(&body2);
    let response2 = router.clone().oneshot(request2).await.unwrap();
    assert_eq!(response2.status(), StatusCode::OK);

    let resp_body2 = collect_body(response2.into_body()).await;
    let json2: serde_json::Value = serde_json::from_slice(&resp_body2).unwrap();
    let content2 = json2["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert!(!content2.is_empty());
    assert!(content2.ends_with(EOS_TEXT_QWEN3));
}

#[tokio::test]
async fn test_reusable_session_slot_reuse_within_timeout() {
    let (router, _manager) = setup_qwen3_fakeecho(1, SessionMode::Reusable, 1, 10, 5000);

    let session_id = "test-reuse-session";

    let body1 = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "First message"}],
        "stream": false,
        "session_id": session_id
    });

    let request1 = chat_request(&body1);
    let response1 = router.clone().oneshot(request1).await.unwrap();
    assert_eq!(response1.status(), StatusCode::OK);

    tokio::time::sleep(Duration::from_millis(50)).await;

    let body2 = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response one"},
            {"role": "user", "content": "Second message"}
        ],
        "stream": false,
        "session_id": session_id
    });

    let request2 = chat_request(&body2);
    let response2 = router.clone().oneshot(request2).await.unwrap();
    assert_eq!(response2.status(), StatusCode::OK);

    let resp_body2 = collect_body(response2.into_body()).await;
    let json2: serde_json::Value = serde_json::from_slice(&resp_body2).unwrap();
    let content2 = json2["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert!(!content2.is_empty());
    assert!(content2.ends_with(EOS_TEXT_QWEN3));
}

// ─── Sequence prefix match (direct) ────────────────────────────────────────

#[tokio::test]
async fn test_reusable_sequence_prefix_match_direct() {
    let (manager, _buffer) = create_test_manager_with_mode(2, 5000, SessionMode::Reusable);

    let session_id = "test-session-direct";

    let messages1 = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
    }];

    let handle1 = manager.acquire_session(session_id).await.unwrap();
    let slot_index = handle1.slot_index;

    let (write_len1, _notifier1) = manager
        .write_prompts(slot_index, session_id, &messages1, None)
        .await
        .unwrap();

    assert!(write_len1 > 0, "first prompt should have tokens");

    let prompt_len1 = manager.get_prompt_length(slot_index);
    assert_eq!(prompt_len1, write_len1);

    let tokens_after_first = manager.slot_sequences.with(|seq| {
        seq.token_ids(slot_index, 0, prompt_len1)
    });
    assert_eq!(tokens_after_first.len(), prompt_len1);

    manager.batch_states.with_mut(|slots| {
        slots[slot_index].phase = Phase::Eos;
        slots[slot_index].next_sequence_index = prompt_len1 + 5;
    });

    let seq_len_after_gen = manager.get_next_sequence_index(slot_index);

    Arc::clone(&manager)
        .release_session(session_id, seq_len_after_gen)
        .await;

    let seq_len_reserved = manager.batch_states.with(|slots| {
        slots[slot_index].sequence_length
    });
    assert_eq!(seq_len_reserved, seq_len_after_gen);
    assert!(seq_len_reserved > 0);

    let tokens_reserved = manager.slot_sequences.with(|seq| {
        seq.token_ids(slot_index, 0, seq_len_reserved)
    });
    assert_eq!(tokens_reserved.len(), seq_len_reserved);
    assert_eq!(&tokens_reserved[..prompt_len1], &tokens_after_first[..]);

    let messages2 = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Hi there".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "How are you?".to_string(),
        },
    ];

    let handle2 = manager.acquire_session(session_id).await.unwrap();
    assert_eq!(handle2.slot_index, slot_index, "should reuse the same slot");

    let (write_len2, _notifier2) = manager
        .write_prompts(slot_index, session_id, &messages2, None)
        .await
        .unwrap();

    let prompt_len2 = manager.get_prompt_length(slot_index);
    assert!(prompt_len2 > prompt_len1, "second prompt should be longer");

    let reused_prefix_len = prompt_len2 - write_len2;
    assert!(
        reused_prefix_len > 0,
        "should have reused prefix tokens: reused_prefix_len={}, write_len2={}, prompt_len2={}",
        reused_prefix_len,
        write_len2,
        prompt_len2
    );

    assert!(
        write_len2 < prompt_len2,
        "write_len should be less than total prompt_len because of prefix reuse"
    );

    let tokens_after_second = manager.slot_sequences.with(|seq| {
        seq.token_ids(slot_index, 0, prompt_len2)
    });

    assert_eq!(
        &tokens_after_second[..reused_prefix_len],
        &tokens_after_first[..reused_prefix_len],
        "reused prefix tokens should be identical (not overwritten)"
    );

    let next_seq_idx = manager.batch_states.with(|slots| {
        slots[slot_index].next_sequence_index
    });
    assert_eq!(
        next_seq_idx, reused_prefix_len,
        "next_sequence_index should start at prefix_len for incremental prefill"
    );
}

// ─── Multi-user multi-round reusable ───────────────────────────────────────

const REUSABLE_NUM_USERS: usize = 50;
const REUSABLE_BATCH_SIZE: usize = 32;
const REUSABLE_NUM_ROUNDS: usize = 3;
const REUSABLE_MAX_RETRIES: usize = 20;
const REUSABLE_RETRY_BASE_DELAY_MS: u64 = 50;

#[tokio::test]
async fn test_multi_user_reusable_multi_round() {
    let (manager, _buffer) = create_qwen3_test_manager_with_mode(
        REUSABLE_BATCH_SIZE,
        5000,
        SessionMode::Reusable,
    );
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let gen_tokens: Vec<usize> = (0..10).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 4, gen_tokens);

    let mut user_handles = Vec::with_capacity(REUSABLE_NUM_USERS);

    for user_id in 0..REUSABLE_NUM_USERS {
        let router = router.clone();
        let initial_delay =
            10 + (user_id as u64 * 37) % 190;

        let handle = tokio::spawn(async move {
            let session_id = format!("reusable-user-{}", user_id);
            let mut round_results = Vec::with_capacity(REUSABLE_NUM_ROUNDS);
            let mut conversation: Vec<(String, String)> = Vec::new();
            let mut total_retries = 0usize;

            for round in 0..REUSABLE_NUM_ROUNDS {
                if round > 0 {
                    let jitter = (user_id as u64 * round as u64 * 13) % 20;
                    tokio::time::sleep(Duration::from_millis(30 + jitter)).await;
                }

                let user_message = format!(
                    "User {} round {}: hello world test message number {}",
                    user_id, round, round + 1
                );

                let mut messages = Vec::new();
                for (role, content) in &conversation {
                    messages.push(serde_json::json!({
                        "role": role,
                        "content": content
                    }));
                }
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": user_message
                }));

                let body = serde_json::json!({
                    "model": "test-model",
                    "messages": messages,
                    "stream": false,
                    "session_id": session_id
                });

                let start = std::time::Instant::now();
                let (response, retries) = send_with_retry(
                    router.clone(),
                    &body,
                    REUSABLE_MAX_RETRIES,
                    REUSABLE_RETRY_BASE_DELAY_MS,
                )
                .await;
                let elapsed = start.elapsed();
                total_retries += retries;

                let resp_body = collect_body(response.into_body()).await;
                let json: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();

                let content = json["choices"][0]["message"]["content"]
                    .as_str()
                    .expect("content should be a string");

                assert!(
                    !content.is_empty(),
                    "user {} round {}: content should not be empty",
                    user_id, round
                );
                assert!(
                    content.ends_with(EOS_TEXT_QWEN3),
                    "user {} round {}: content should end with eos, got: {:?}",
                    user_id, round, content
                );

                let generated_part = &content[..content.len() - EOS_TEXT_QWEN3.len()];
                assert!(
                    !generated_part.is_empty(),
                    "user {} round {}: should have generated tokens before eos",
                    user_id, round
                );

                let expected_pattern = "0123456789".repeat(generated_part.len() / 10 + 1);
                assert!(
                    generated_part
                        .chars()
                        .eq(expected_pattern.chars().take(generated_part.len())),
                    "user {} round {}: generated content should cycle through 0-9 digits, got: {:?}",
                    user_id, round, generated_part
                );

                conversation.push(("user".to_string(), user_message));
                conversation.push((
                    "assistant".to_string(),
                    generated_part.to_string(),
                ));

                round_results.push((round, elapsed, generated_part.len(), retries));
            }

            (user_id, round_results, total_retries)
        });

        user_handles.push(handle);
        tokio::time::sleep(Duration::from_millis(initial_delay)).await;
    }

    let mut all_results = Vec::with_capacity(REUSABLE_NUM_USERS);
    for handle in user_handles {
        let (user_id, round_results, total_retries) = handle.await.unwrap();
        all_results.push((user_id, round_results, total_retries));
    }

    assert_eq!(all_results.len(), REUSABLE_NUM_USERS);

    let mut total_requests = 0;
    let mut total_latency = Duration::from_millis(0);
    let mut max_latency = Duration::from_millis(0);
    let mut min_latency = Duration::from_millis(u64::MAX);
    let mut total_gen_len = 0usize;
    let mut total_retries_all = 0usize;
    let mut users_with_retries = 0usize;

    for (user_id, rounds, total_retries) in &all_results {
        for (round, latency, gen_len, _retries) in rounds {
            total_requests += 1;
            total_latency += *latency;
            if *latency > max_latency {
                max_latency = *latency;
            }
            if *latency < min_latency {
                min_latency = *latency;
            }
            total_gen_len += gen_len;
        }
        assert_eq!(
            rounds.len(),
            REUSABLE_NUM_ROUNDS,
            "user {} should have {} rounds",
            user_id,
            REUSABLE_NUM_ROUNDS
        );
        total_retries_all += total_retries;
        if *total_retries > 0 {
            users_with_retries += 1;
        }
    }

    let avg_latency = total_latency / total_requests as u32;
    let avg_gen_len = total_gen_len as f64 / total_requests as f64;

    println!(
        "Reusable multi-user multi-round test: {} users, batch_size={}, {} rounds each, total_requests={}",
        REUSABLE_NUM_USERS, REUSABLE_BATCH_SIZE, REUSABLE_NUM_ROUNDS, total_requests
    );
    println!(
        "  avg_latency={:?}, min_latency={:?}, max_latency={:?}",
        avg_latency, min_latency, max_latency
    );
    println!("  avg_gen_len={:.1}", avg_gen_len);
    println!(
        "  total_retries={}, users_with_retries={}/{}",
        total_retries_all, users_with_retries, REUSABLE_NUM_USERS
    );
}
