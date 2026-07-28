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
use crate::serving::server::build_router;

use super::test_utils::*;

fn parse_sse_events(body: &str) -> Vec<serde_json::Value> {
    let mut events = Vec::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                events.push(json);
            }
        }
    }
    events
}

#[tokio::test]
async fn test_stream_chat_completions_with_fakeecho_parser_false() {
    let (manager, _buffer) =
        create_qwen3_test_manager_with_parser(4, 1000, SessionMode::NonReusable, false, false);
    let router = build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, digit_tokens, 10);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": true
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("text/event-stream"));

    let resp_body = collect_body(response.into_body()).await;
    let body_str = String::from_utf8(resp_body).unwrap();

    assert!(body_str.contains("data:"));
    assert!(body_str.contains("chat.completion.chunk"));

    let events = parse_sse_events(&body_str);
    assert!(!events.is_empty(), "should have at least one SSE event");

    let mut full_content = String::new();
    let mut has_role = false;
    let mut finish_event_count = 0;

    for event in &events {
        assert_eq!(event["object"], "chat.completion.chunk");
        assert_eq!(event["model"], "test-model");
        assert!(event["id"].is_string());
        assert!(event["created"].is_number());
        assert!(event["choices"].is_array());
        assert_eq!(event["choices"][0]["index"], 0);

        let delta = &event["choices"][0]["delta"];

        if let Some(role) = delta["role"].as_str() {
            assert_eq!(role, "assistant");
            has_role = true;
        }

        if let Some(content) = delta["content"].as_str() {
            full_content.push_str(content);
        }

        if event["choices"][0]["finish_reason"] == "stop" {
            finish_event_count += 1;
        }
    }

    assert!(has_role, "first event should have role: assistant");
    assert_eq!(
        finish_event_count, 1,
        "should have exactly one finish event"
    );
    assert!(
        !full_content.is_empty(),
        "generated content should not be empty"
    );

    let eos_text = "<|im_end|>";
    assert!(
        full_content.ends_with(eos_text),
        "content should end with eos token <|im_end|>, got: {:?}",
        full_content
    );

    let generated_part = &full_content[..full_content.len() - eos_text.len()];
    assert!(
        !generated_part.is_empty(),
        "should have generated tokens before eos"
    );

    let expected_pattern = "0123456789".repeat(generated_part.len() / 10 + 1);
    assert!(
        generated_part
            .chars()
            .eq(expected_pattern.chars().take(generated_part.len())),
        "generated content should cycle through 0-9 digits, got: {:?}",
        generated_part
    );
}

const STREAM_NUM_USERS: usize = 10;
const STREAM_BATCH_SIZE: usize = 16;
const STREAM_MIN_DELAY_MS: u64 = 50;
const STREAM_MAX_DELAY_MS: u64 = 300;
const STREAM_INTER_REQUEST_DELAY_MS: u64 = 100;

#[tokio::test]
async fn test_stream_multi_user_with_fakeecho_parser_false() {
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        STREAM_BATCH_SIZE,
        5000,
        SessionMode::NonReusable,
        false,
        false,
    );
    let router = build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, digit_tokens, 10);

    let mut handles = Vec::with_capacity(STREAM_NUM_USERS);

    for user_id in 0..STREAM_NUM_USERS {
        let router = router.clone();
        let arrival_delay = STREAM_MIN_DELAY_MS
            + (user_id as u64 * 23) % (STREAM_MAX_DELAY_MS - STREAM_MIN_DELAY_MS);

        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(arrival_delay)).await;

            let body = serde_json::json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": format!("Hello from user {}", user_id)}
                ],
                "stream": true
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
            let content_type = response.headers().get("content-type").unwrap();
            assert!(content_type.to_str().unwrap().contains("text/event-stream"));

            let resp_body = collect_body(response.into_body()).await;
            let body_str = String::from_utf8(resp_body).unwrap();

            let events = parse_sse_events(&body_str);
            assert!(
                !events.is_empty(),
                "user {}: should have at least one SSE event",
                user_id
            );

            let mut full_content = String::new();
            let mut finish_count = 0;

            for event in &events {
                assert_eq!(event["object"], "chat.completion.chunk", "user {}", user_id);

                let delta = &event["choices"][0]["delta"];
                if let Some(content) = delta["content"].as_str() {
                    full_content.push_str(content);
                }

                if event["choices"][0]["finish_reason"] == "stop" {
                    finish_count += 1;
                }
            }

            assert_eq!(
                finish_count, 1,
                "user {}: should have exactly one finish event",
                user_id
            );

            assert!(
                !full_content.is_empty(),
                "user {}: content should not be empty",
                user_id
            );

            let eos_text = "<|im_end|>";
            assert!(
                full_content.ends_with(eos_text),
                "user {}: content should end with eos token, got: {:?}",
                user_id,
                full_content
            );

            let generated_part = &full_content[..full_content.len() - eos_text.len()];
            assert!(
                !generated_part.is_empty(),
                "user {}: should have generated tokens before eos",
                user_id
            );

            let expected_pattern = "0123456789".repeat(generated_part.len() / 10 + 1);
            assert!(
                generated_part
                    .chars()
                    .eq(expected_pattern.chars().take(generated_part.len())),
                "user {}: generated content should cycle through 0-9 digits, got: {:?}",
                user_id,
                generated_part
            );

            (user_id, elapsed, generated_part.len())
        });

        handles.push(handle);
    }

    let mut results = Vec::with_capacity(STREAM_NUM_USERS);
    for handle in handles {
        let (user_id, elapsed, gen_len) = handle.await.unwrap();
        results.push((user_id, elapsed, gen_len));
    }

    assert_eq!(results.len(), STREAM_NUM_USERS);

    let total_time: Duration = results.iter().map(|(_, e, _)| *e).sum();
    let avg_time = total_time / STREAM_NUM_USERS as u32;
    let max_time = results.iter().map(|(_, e, _)| *e).max().unwrap();
    let min_time = results.iter().map(|(_, e, _)| *e).min().unwrap();
    let avg_gen_len: f64 =
        results.iter().map(|(_, _, l)| *l as f64).sum::<f64>() / STREAM_NUM_USERS as f64;

    println!(
        "Stream multi-user test (parser=false, qwen3): {} users, batch_size={}, avg_latency={:?}, min_latency={:?}, max_latency={:?}, avg_gen_len={:.1}",
        STREAM_NUM_USERS, STREAM_BATCH_SIZE, avg_time, min_time, max_time, avg_gen_len
    );
}

#[tokio::test]
async fn test_stream_burst_traffic_with_fakeecho_parser_false() {
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        STREAM_BATCH_SIZE,
        5000,
        SessionMode::NonReusable,
        false,
        false,
    );
    let router = build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, digit_tokens, 10);

    let mut handles = Vec::with_capacity(STREAM_NUM_USERS);

    for burst_id in 0..2 {
        for i in 0..STREAM_NUM_USERS / 2 {
            let user_id = burst_id * (STREAM_NUM_USERS / 2) + i;
            let router = router.clone();
            let burst_delay = burst_id as u64 * STREAM_INTER_REQUEST_DELAY_MS;
            let jitter = (i as u64 * 17) % 50;

            let handle = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(burst_delay + jitter)).await;

                let body = serde_json::json!({
                    "model": "test-model",
                    "messages": [
                        {"role": "user", "content": format!("Burst {} user {}", burst_id, user_id)}
                    ],
                    "stream": true
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
                let body_str = String::from_utf8(resp_body).unwrap();

                let events = parse_sse_events(&body_str);
                assert!(
                    !events.is_empty(),
                    "burst user {}: should have SSE events",
                    user_id
                );

                let mut full_content = String::new();
                for event in &events {
                    let delta = &event["choices"][0]["delta"];
                    if let Some(content) = delta["content"].as_str() {
                        full_content.push_str(content);
                    }
                }

                let eos_text = "<|im_end|>";
                assert!(
                    full_content.ends_with(eos_text),
                    "burst user {}: content should end with eos",
                    user_id
                );

                (user_id, elapsed)
            });

            handles.push(handle);
        }
    }

    let mut results = Vec::with_capacity(STREAM_NUM_USERS);
    for handle in handles {
        let (user_id, elapsed) = handle.await.unwrap();
        results.push((user_id, elapsed));
    }

    assert_eq!(results.len(), STREAM_NUM_USERS);

    let total_time: Duration = results.iter().map(|(_, e)| *e).sum();
    let avg_time = total_time / STREAM_NUM_USERS as u32;
    let max_time = results.iter().map(|(_, e)| *e).max().unwrap();
    let min_time = results.iter().map(|(_, e)| *e).min().unwrap();

    println!(
        "Stream burst traffic test (parser=false, qwen3): {} users in 2 bursts, batch_size={}, avg_latency={:?}, min_latency={:?}, max_latency={:?}",
        STREAM_NUM_USERS, STREAM_BATCH_SIZE, avg_time, min_time, max_time
    );
}

#[tokio::test]
async fn test_stream_overload_returns_503_service_unavailable() {
    let batch_size = 2;
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        batch_size,
        5000,
        SessionMode::NonReusable,
        false,
        false,
    );
    let router = build_router(Arc::clone(&manager));

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
        "stream": true
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
async fn test_stream_overload_recovery_after_slot_release() {
    let batch_size = 2;
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        batch_size,
        5000,
        SessionMode::NonReusable,
        false,
        false,
    );
    let router = build_router(Arc::clone(&manager));

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
        "stream": true,
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
        "stream": true,
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

    let content_type = response2.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("text/event-stream"));
}

fn start_runtime_with_fakeecho(
    manager: Arc<SlotManager<f16>>,
    eos_id: usize,
    thread_num: usize,
    tokens: Vec<usize>,
    max_gen_tokens: usize,
) -> Arc<Scheduler> {
    let (batch_size, seq_len, sequences_ptr) = manager
        .batch_sequences
        .with(|seq| (seq.row_size, seq.col_size, seq.sequences));

    let batch_states = manager.batch_states.clone();
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        seq_len,
        thread_num,
        batch_states,
    ));

    let fake_echo = FakeEcho::new(sequences_ptr, seq_len, eos_id, tokens, max_gen_tokens);
    let operator_queue: Vec<Operator<f16>> = vec![Operator::FakeEcho(fake_echo)];

    let executor = ExecutorPool::new(operator_queue, Arc::clone(&scheduler), thread_num);
    executor.start();

    scheduler
}
