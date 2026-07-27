use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::operators::fake_echo::FakeEcho;
use crate::operators::operator::Operator;
use crate::runtime::executor::executor_pool::ExecutorPool;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{SessionMode, SlotManager};

use super::test_utils::*;

#[tokio::test]
async fn test_chat_completions_with_scheduler_fakeecho() {
    let (router, manager, _buffer) = create_qwen3_test_router_with_mode(SessionMode::NonReusable);

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, digit_tokens, 10);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": false
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let resp_body = collect_body(response.into_body()).await;
    let json: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "test-model");
    assert!(json["id"].is_string());
    assert!(json["created"].is_number());
    assert!(json["choices"].is_array());
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert!(!content.is_empty(), "generated content should not be empty");

    let eos_text = "<|im_end|>";
    assert!(
        content.ends_with(eos_text),
        "content should end with eos token <|im_end|>, got: {:?}",
        content
    );

    let generated_part = &content[..content.len() - eos_text.len()];
    assert!(
        !generated_part.is_empty(),
        "should have generated tokens before eos"
    );

    let expected_pattern = "0123456789".repeat(generated_part.len() / 10 + 1);
    assert!(
        generated_part.chars().eq(expected_pattern.chars().take(generated_part.len())),
        "generated content should cycle through 0-9 digits, got: {:?}",
        generated_part
    );
}

const NUM_USERS: usize = 10;
const BATCH_SIZE: usize = 16;
const MIN_DELAY_MS: u64 = 50;
const MAX_DELAY_MS: u64 = 300;
const INTER_REQUEST_DELAY_MS: u64 = 100;

#[tokio::test]
async fn test_multi_user_with_scheduler_fakeecho() {
    let (manager, _buffer) =
        create_qwen3_test_manager_with_mode(BATCH_SIZE, 5000, SessionMode::NonReusable);
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, digit_tokens, 10);

    let mut handles = Vec::with_capacity(NUM_USERS);

    for user_id in 0..NUM_USERS {
        let router = router.clone();
        let arrival_delay = MIN_DELAY_MS + (user_id as u64 * 23) % (MAX_DELAY_MS - MIN_DELAY_MS);

        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(arrival_delay)).await;

            let body = serde_json::json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": format!("Hello from user {}", user_id)}
                ],
                "stream": false
            });

            let request = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();

            let start = std::time::Instant::now();
            let response = router.oneshot(request).await.unwrap();
            let elapsed = start.elapsed();

            assert_eq!(response.status(), StatusCode::OK);

            let resp_body = collect_body(response.into_body()).await;
            let json: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();

            assert_eq!(json["object"], "chat.completion");
            assert_eq!(json["model"], "test-model");
            assert!(json["id"].is_string());
            assert!(json["created"].is_number());
            assert!(json["choices"].is_array());
            assert_eq!(json["choices"][0]["index"], 0);
            assert_eq!(json["choices"][0]["message"]["role"], "assistant");
            assert_eq!(json["choices"][0]["finish_reason"], "stop");

            let content = json["choices"][0]["message"]["content"]
                .as_str()
                .expect("content should be a string");
            assert!(!content.is_empty(), "user {}: content should not be empty", user_id);

            let eos_text = "<|im_end|>";
            assert!(
                content.ends_with(eos_text),
                "user {}: content should end with eos token, got: {:?}",
                user_id,
                content
            );

            let generated_part = &content[..content.len() - eos_text.len()];
            assert!(
                !generated_part.is_empty(),
                "user {}: should have generated tokens before eos",
                user_id
            );

            let expected_pattern = "0123456789".repeat(generated_part.len() / 10 + 1);
            assert!(
                generated_part.chars().eq(expected_pattern.chars().take(generated_part.len())),
                "user {}: generated content should cycle through 0-9 digits, got: {:?}",
                user_id,
                generated_part
            );

            (user_id, elapsed, generated_part.len())
        });

        handles.push(handle);
    }

    let mut results = Vec::with_capacity(NUM_USERS);
    for handle in handles {
        let (user_id, elapsed, gen_len) = handle.await.unwrap();
        results.push((user_id, elapsed, gen_len));
    }

    assert_eq!(results.len(), NUM_USERS);

    let total_time: Duration = results.iter().map(|(_, e, _)| *e).sum();
    let avg_time = total_time / NUM_USERS as u32;
    let max_time = results.iter().map(|(_, e, _)| *e).max().unwrap();
    let min_time = results.iter().map(|(_, e, _)| *e).min().unwrap();
    let avg_gen_len: f64 = results.iter().map(|(_, _, l)| *l as f64).sum::<f64>() / NUM_USERS as f64;

    println!(
        "Multi-user test (qwen3): {} users, batch_size={}, avg_latency={:?}, min_latency={:?}, max_latency={:?}, avg_gen_len={:.1}",
        NUM_USERS, BATCH_SIZE, avg_time, min_time, max_time, avg_gen_len
    );
}

#[tokio::test]
async fn test_burst_traffic_with_scheduler_fakeecho() {
    let (manager, _buffer) =
        create_qwen3_test_manager_with_mode(BATCH_SIZE, 5000, SessionMode::NonReusable);
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, digit_tokens, 10);

    let mut handles = Vec::with_capacity(NUM_USERS);

    for burst_id in 0..2 {
        for i in 0..NUM_USERS / 2 {
            let user_id = burst_id * (NUM_USERS / 2) + i;
            let router = router.clone();
            let burst_delay = burst_id as u64 * INTER_REQUEST_DELAY_MS;
            let jitter = (i as u64 * 17) % 50;

            let handle = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(burst_delay + jitter)).await;

                let body = serde_json::json!({
                    "model": "test-model",
                    "messages": [
                        {"role": "user", "content": format!("Burst {} user {}", burst_id, user_id)}
                    ],
                    "stream": false
                });

                let request = Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("Content-Type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap();

                let start = std::time::Instant::now();
                let response = router.oneshot(request).await.unwrap();
                let elapsed = start.elapsed();

                assert_eq!(response.status(), StatusCode::OK);

                let resp_body = collect_body(response.into_body()).await;
                let json: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();

                let content = json["choices"][0]["message"]["content"]
                    .as_str()
                    .expect("content should be a string");

                let eos_text = "<|im_end|>";
                assert!(content.ends_with(eos_text));

                (user_id, elapsed)
            });

            handles.push(handle);
        }
    }

    let mut results = Vec::with_capacity(NUM_USERS);
    for handle in handles {
        let (user_id, elapsed) = handle.await.unwrap();
        results.push((user_id, elapsed));
    }

    assert_eq!(results.len(), NUM_USERS);

    let total_time: Duration = results.iter().map(|(_, e)| *e).sum();
    let avg_time = total_time / NUM_USERS as u32;
    let max_time = results.iter().map(|(_, e)| *e).max().unwrap();
    let min_time = results.iter().map(|(_, e)| *e).min().unwrap();

    println!(
        "Burst traffic test (qwen3): {} users in 2 bursts, batch_size={}, avg_latency={:?}, min_latency={:?}, max_latency={:?}",
        NUM_USERS, BATCH_SIZE, avg_time, min_time, max_time
    );
}

#[tokio::test]
async fn test_overload_returns_503_service_unavailable() {
    let batch_size = 2;
    let (manager, _buffer) =
        create_qwen3_test_manager_with_mode(batch_size, 5000, SessionMode::NonReusable);
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, digit_tokens, 10);

    let _h1 = manager.acquire_session("active_user_1").await.unwrap();
    let _h2 = manager.acquire_session("active_user_2").await.unwrap();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello overload"}
        ],
        "stream": false
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

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
async fn test_overload_recovery_after_slot_release() {
    let batch_size = 2;
    let (manager, _buffer) =
        create_qwen3_test_manager_with_mode(batch_size, 5000, SessionMode::NonReusable);
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, digit_tokens, 10);

    let _h1 = manager.acquire_session("held_user_1").await.unwrap();
    let _h2 = manager.acquire_session("held_user_2").await.unwrap();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Should fail"}
        ],
        "stream": false,
        "session_id": "new_user_1"
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = router.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    drop(_h1);
    drop(_h2);
    manager.release_session("held_user_1", 0).await;

    let body2 = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Should succeed now"}
        ],
        "stream": false,
        "session_id": "new_user_2"
    });

    let request2 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body2).unwrap()))
        .unwrap();

    let response2 = router.clone().oneshot(request2).await.unwrap();
    assert_eq!(response2.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_reusable_session_two_requests_same_session() {
    let (router, manager, _buffer) = create_qwen3_test_router_with_mode(SessionMode::Reusable);

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, digit_tokens, 10);

    let session_id = "reusable-session-1";

    let body1 = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": false,
        "session_id": session_id
    });

    let request1 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body1).unwrap()))
        .unwrap();

    let response1 = router.clone().oneshot(request1).await.unwrap();
    assert_eq!(response1.status(), StatusCode::OK);

    let resp_body1 = collect_body(response1.into_body()).await;
    let json1: serde_json::Value = serde_json::from_slice(&resp_body1).unwrap();
    let content1 = json1["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert!(!content1.is_empty());
    assert!(content1.ends_with("<|im_end|>"));

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

    let request2 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body2).unwrap()))
        .unwrap();

    let response2 = router.clone().oneshot(request2).await.unwrap();
    assert_eq!(response2.status(), StatusCode::OK);

    let resp_body2 = collect_body(response2.into_body()).await;
    let json2: serde_json::Value = serde_json::from_slice(&resp_body2).unwrap();
    let content2 = json2["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert!(!content2.is_empty());
    assert!(content2.ends_with("<|im_end|>"));
}

#[tokio::test]
async fn test_reusable_session_slot_reuse_within_timeout() {
    let batch_size = 1;
    let (manager, _buffer) =
        create_qwen3_test_manager_with_mode(batch_size, 5000, SessionMode::Reusable);
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, digit_tokens, 10);

    let session_id = "test-reuse-session";

    let body1 = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "First message"}
        ],
        "stream": false,
        "session_id": session_id
    });

    let request1 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body1).unwrap()))
        .unwrap();

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

    let request2 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body2).unwrap()))
        .unwrap();

    let response2 = router.clone().oneshot(request2).await.unwrap();
    assert_eq!(response2.status(), StatusCode::OK);

    let resp_body2 = collect_body(response2.into_body()).await;
    let json2: serde_json::Value = serde_json::from_slice(&resp_body2).unwrap();
    let content2 = json2["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert!(!content2.is_empty());
    assert!(content2.ends_with("<|im_end|>"));
}

#[tokio::test]
async fn test_reusable_sequence_prefix_match_direct() {
    use crate::runtime::session::Phase;
    use crate::serving::ChatMessage;

    let (manager, _buffer) = create_test_manager_with_mode(2, 5000, SessionMode::Reusable);

    let session_id = "test-session-direct";

    let messages1 = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        },
    ];

    let handle1 = manager.acquire_session(session_id).await.unwrap();
    let slot_index = handle1.slot_index;

    let (write_len1, _notifier1) = manager
        .write_prompts(slot_index, session_id, &messages1, None)
        .await
        .unwrap();

    assert!(write_len1 > 0, "first prompt should have tokens");

    let prompt_len1 = manager.get_prompt_length(slot_index);
    assert_eq!(prompt_len1, write_len1);

    let tokens_after_first = manager.batch_sequences.with(|seq| {
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

    let tokens_reserved = manager.batch_sequences.with(|seq| {
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

    let tokens_after_second = manager.batch_sequences.with(|seq| {
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

fn start_runtime_with_fakeecho(
    manager: Arc<SlotManager<f16>>,
    eos_id: usize,
    thread_num: usize,
    tokens: Vec<usize>,
    max_gen_tokens: usize,
) -> Arc<Scheduler> {
    let (batch_size, seq_len, sequences_ptr) = manager.batch_sequences.with(|seq| {
        (seq.row_size, seq.col_size, seq.sequences)
    });

    let batch_states = manager.batch_states.clone();
    let scheduler = Arc::new(Scheduler::new(batch_size, seq_len, thread_num, batch_states));

    let fake_echo = FakeEcho::new(sequences_ptr, seq_len, eos_id, tokens, max_gen_tokens);
    let operator_queue: Vec<Operator<f16>> = vec![Operator::FakeEcho(fake_echo)];

    let executor = ExecutorPool::new(operator_queue, Arc::clone(&scheduler), thread_num);
    executor.start();

    scheduler
}
