use axum::http::StatusCode;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::runtime::session::SessionMode;

use super::test_utils::*;

const EOS_TEXT_QWEN3: &str = "<|im_end|>";
const EOS_TEXT_R50K: &str = "<|endoftext|>";

// ─── Helpers ───────────────────────────────────────────────────────────────

async fn run_sync_request(router: axum::Router, user_id: usize, eos_text: &str) -> Duration {
    let body = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": format!("Hello from user {}", user_id)}
        ],
        "stream": false
    });

    let request = chat_request(&body);
    let start = std::time::Instant::now();
    let response = router.oneshot(request).await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(response.status(), StatusCode::OK);

    let resp_body = collect_body(response.into_body()).await;
    let json: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();

    assert_sync_response_basic(&json);
    let generated = assert_sync_content_ends_with(&json, eos_text);
    assert_digit_pattern(&generated);

    elapsed
}

async fn run_stream_request(router: axum::Router, user_id: usize, eos_text: &str) -> Duration {
    let body = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": format!("Hello from user {}", user_id)}
        ],
        "stream": true
    });

    let request = chat_request(&body);
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

    let result = collect_stream_result(&events);
    assert_eq!(
        result.finish_event_count, 1,
        "user {}: should have exactly one finish event",
        user_id
    );
    assert!(
        result.full_content.ends_with(eos_text),
        "user {}: content should end with eos, got: {:?}",
        user_id,
        result.full_content
    );

    let generated = &result.full_content[..result.full_content.len() - eos_text.len()];
    assert_digit_pattern(generated);

    elapsed
}

fn print_latency_stats(label: &str, results: &[(usize, Duration)]) {
    let n = results.len();
    let total: Duration = results.iter().map(|(_, e)| *e).sum();
    let avg = total / n as u32;
    let max = results.iter().map(|(_, e)| *e).max().unwrap();
    let min = results.iter().map(|(_, e)| *e).min().unwrap();
    println!(
        "{}: {} users, avg={:?}, min={:?}, max={:?}",
        label, n, avg, min, max
    );
}

// ─── Sync concurrent (r50k) ────────────────────────────────────────────────

const SYNC_R50K_USERS: usize = 20;
const SYNC_R50K_BATCH: usize = 32;

#[tokio::test]
async fn test_concurrent_sync_r50k() {
    let (manager, _buffer) =
        create_test_manager_with_mode(SYNC_R50K_BATCH, 1000, SessionMode::NonReusable);
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let digit_tokens = digit_tokens_r50k();
    let eos_id = 50256;
    let gen_tokens: Vec<usize> = (0..10).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, gen_tokens);

    let mut handles = Vec::with_capacity(SYNC_R50K_USERS);

    for user_id in 0..SYNC_R50K_USERS {
        let router = router.clone();
        let delay = 50 + (user_id as u64 * 7) % 150;

        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            let elapsed = run_sync_request(router, user_id, EOS_TEXT_R50K).await;
            (user_id, elapsed)
        });

        handles.push(handle);
    }

    let mut results = Vec::with_capacity(SYNC_R50K_USERS);
    for handle in handles {
        let (user_id, elapsed) = handle.await.unwrap();
        results.push((user_id, elapsed));
    }

    assert_eq!(results.len(), SYNC_R50K_USERS);
    print_latency_stats("Sync concurrent (r50k)", &results);
}

// ─── Sync concurrent (qwen3) ───────────────────────────────────────────────

const SYNC_QWEN3_USERS: usize = 10;
const SYNC_QWEN3_BATCH: usize = 16;

#[tokio::test]
async fn test_concurrent_sync_qwen3() {
    let (manager, _buffer) =
        create_qwen3_test_manager_with_mode(SYNC_QWEN3_BATCH, 5000, SessionMode::NonReusable);
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let gen_tokens: Vec<usize> = (0..10).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, gen_tokens);

    let mut handles = Vec::with_capacity(SYNC_QWEN3_USERS);

    for user_id in 0..SYNC_QWEN3_USERS {
        let router = router.clone();
        let delay = 50 + (user_id as u64 * 23) % 250;

        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            let elapsed = run_sync_request(router, user_id, EOS_TEXT_QWEN3).await;
            (user_id, elapsed)
        });

        handles.push(handle);
    }

    let mut results = Vec::with_capacity(SYNC_QWEN3_USERS);
    for handle in handles {
        let (user_id, elapsed) = handle.await.unwrap();
        results.push((user_id, elapsed));
    }

    assert_eq!(results.len(), SYNC_QWEN3_USERS);
    print_latency_stats("Sync concurrent (qwen3)", &results);
}

// ─── Stream concurrent (qwen3, parser=false) ───────────────────────────────

const STREAM_QWEN3_USERS: usize = 10;
const STREAM_QWEN3_BATCH: usize = 16;

#[tokio::test]
async fn test_concurrent_stream_qwen3_parser_false() {
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        STREAM_QWEN3_BATCH,
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
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, gen_tokens);

    let mut handles = Vec::with_capacity(STREAM_QWEN3_USERS);

    for user_id in 0..STREAM_QWEN3_USERS {
        let router = router.clone();
        let delay = 50 + (user_id as u64 * 23) % 250;

        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            let elapsed = run_stream_request(router, user_id, EOS_TEXT_QWEN3).await;
            (user_id, elapsed)
        });

        handles.push(handle);
    }

    let mut results = Vec::with_capacity(STREAM_QWEN3_USERS);
    for handle in handles {
        let (user_id, elapsed) = handle.await.unwrap();
        results.push((user_id, elapsed));
    }

    assert_eq!(results.len(), STREAM_QWEN3_USERS);
    print_latency_stats("Stream concurrent (qwen3, parser=false)", &results);
}

// ─── Burst traffic (sync, qwen3) ───────────────────────────────────────────

const BURST_USERS: usize = 10;
const BURST_BATCH: usize = 16;
const BURST_INTERVAL_MS: u64 = 100;

#[tokio::test]
async fn test_burst_sync_qwen3() {
    let (manager, _buffer) =
        create_qwen3_test_manager_with_mode(BURST_BATCH, 5000, SessionMode::NonReusable);
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let gen_tokens: Vec<usize> = (0..10).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, gen_tokens);

    let mut handles = Vec::with_capacity(BURST_USERS);

    for burst_id in 0..2 {
        for i in 0..BURST_USERS / 2 {
            let user_id = burst_id * (BURST_USERS / 2) + i;
            let router = router.clone();
            let burst_delay = burst_id as u64 * BURST_INTERVAL_MS;
            let jitter = (i as u64 * 17) % 50;

            let handle = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(burst_delay + jitter)).await;
                let elapsed = run_sync_request(router, user_id, EOS_TEXT_QWEN3).await;
                (user_id, elapsed)
            });

            handles.push(handle);
        }
    }

    let mut results = Vec::with_capacity(BURST_USERS);
    for handle in handles {
        let (user_id, elapsed) = handle.await.unwrap();
        results.push((user_id, elapsed));
    }

    assert_eq!(results.len(), BURST_USERS);
    print_latency_stats("Burst sync (qwen3, 2 bursts)", &results);
}

// ─── Burst traffic (stream, qwen3, parser=false) ───────────────────────────

#[tokio::test]
async fn test_burst_stream_qwen3_parser_false() {
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        BURST_BATCH,
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
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, gen_tokens);

    let mut handles = Vec::with_capacity(BURST_USERS);

    for burst_id in 0..2 {
        for i in 0..BURST_USERS / 2 {
            let user_id = burst_id * (BURST_USERS / 2) + i;
            let router = router.clone();
            let burst_delay = burst_id as u64 * BURST_INTERVAL_MS;
            let jitter = (i as u64 * 17) % 50;

            let handle = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(burst_delay + jitter)).await;
                let elapsed = run_stream_request(router, user_id, EOS_TEXT_QWEN3).await;
                (user_id, elapsed)
            });

            handles.push(handle);
        }
    }

    let mut results = Vec::with_capacity(BURST_USERS);
    for handle in handles {
        let (user_id, elapsed) = handle.await.unwrap();
        results.push((user_id, elapsed));
    }

    assert_eq!(results.len(), BURST_USERS);
    print_latency_stats("Burst stream (qwen3, parser=false, 2 bursts)", &results);
}

// ─── Stream multi-user (parser=true, FakeParserEcho) ───────────────────────

const PARSER_TRUE_USERS: usize = 8;
const PARSER_TRUE_BATCH: usize = 16;

#[tokio::test]
async fn test_concurrent_stream_qwen3_parser_true() {
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        PARSER_TRUE_BATCH,
        10000,
        SessionMode::NonReusable,
        true,
        true,
    );
    let router = crate::serving::server::build_router(Arc::clone(&manager));

    let script_text = concat!(
        "Hi there!",
        "<think>Let me process this request carefully.</think>",
        "I'll use a tool to help you.",
        "<tool_call>{\"name\":\"calculator\",\"arguments\":{\"expression\":\"2+2\"}}</tool_call>",
        "Done processing."
    );

    let tokenizer = qwen3_tokenizer();
    let eos_id = 151645;
    let script_tokens = tokenizer
        .encode_with_special_tokens(script_text)
        .into_iter()
        .map(|t| t as usize)
        .collect::<Vec<_>>();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 2, script_tokens);

    let mut handles = Vec::with_capacity(PARSER_TRUE_USERS);

    for user_id in 0..PARSER_TRUE_USERS {
        let router = router.clone();
        let delay = 50 + (user_id as u64 * 23) % 250;

        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;

            let body = serde_json::json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": format!("Hi from user {}", user_id)}
                ],
                "stream": true
            });

            let request = chat_request(&body);
            let start = std::time::Instant::now();
            let response = router.oneshot(request).await.unwrap();
            let elapsed = start.elapsed();

            assert_eq!(response.status(), StatusCode::OK);

            let resp_body = collect_body(response.into_body()).await;
            let body_str = String::from_utf8(resp_body).unwrap();
            let events = parse_sse_events(&body_str);
            assert!(
                !events.is_empty(),
                "user {}: should have at least one SSE event",
                user_id
            );

            let result = collect_stream_result(&events);
            assert_eq!(
                result.finish_event_count, 1,
                "user {}: should have exactly one finish event",
                user_id
            );
            assert!(
                !result.full_content.is_empty(),
                "user {}: content should not be empty",
                user_id
            );
            assert!(
                result.has_reasoning,
                "user {}: should have reasoning content",
                user_id
            );
            assert!(
                result.has_tool_calls,
                "user {}: should have tool calls",
                user_id
            );
            assert!(
                result.full_reasoning.contains("process this request"),
                "user {}: reasoning content mismatch, got: {:?}",
                user_id,
                result.full_reasoning
            );

            (user_id, elapsed)
        });

        handles.push(handle);
    }

    let mut results = Vec::with_capacity(PARSER_TRUE_USERS);
    for handle in handles {
        let (user_id, elapsed) = handle.await.unwrap();
        results.push((user_id, elapsed));
    }

    assert_eq!(results.len(), PARSER_TRUE_USERS);
    print_latency_stats("Stream concurrent (qwen3, parser=true)", &results);
}
