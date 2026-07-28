use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;

use crate::runtime::session::SessionMode;
use crate::serving::server::build_router;

use super::test_utils::*;

// ─── Status endpoint ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_status_endpoint() {
    let (router, _manager, _buffer) = create_test_router();

    let request = Request::builder()
        .method("GET")
        .uri("/status")
        .body(Body::empty())
        .unwrap();

    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = collect_body(response.into_body()).await;
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "running");
    assert_eq!(json["mode"], "inlined_scheduler");
    assert!(json["info"].is_string());
}

// ─── Error cases ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_chat_completions_invalid_json() {
    let (router, _manager, _buffer) = create_test_router();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from("not valid json {{{"))
        .unwrap();

    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_completions_missing_fields() {
    let (router, _manager, _buffer) = create_test_router();

    let body = serde_json::json!({
        "model": "test-model"
    });

    let request = chat_request(&body);
    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// ─── Sync full flow (r50k tokenizer + FakeEcho) ────────────────────────────

#[tokio::test]
async fn test_sync_full_flow_r50k() {
    let (manager, _buffer) = create_test_manager_with_mode(4, 1000, SessionMode::NonReusable);
    let router = build_router(Arc::clone(&manager));

    let digit_tokens = digit_tokens_r50k();
    let eos_id = 50256;
    let gen_tokens: Vec<usize> = (0..10).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, gen_tokens);

    let body = simple_chat_body("Hello", false);
    let request = chat_request(&body);
    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let resp_body = collect_body(response.into_body()).await;
    let json: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();

    assert_sync_response_basic(&json);

    let eos_text = "<|endoftext|>";
    let generated = assert_sync_content_ends_with(&json, eos_text);
    assert_digit_pattern(&generated);
}

// ─── Stream full flow (r50k tokenizer + FakeEcho) ──────────────────────────

#[tokio::test]
async fn test_stream_full_flow_r50k() {
    let (manager, _buffer) = create_test_manager_with_mode(4, 1000, SessionMode::NonReusable);
    let router = build_router(Arc::clone(&manager));

    let digit_tokens = digit_tokens_r50k();
    let eos_id = 50256;
    let gen_tokens: Vec<usize> = (0..10).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1, gen_tokens);

    let body = simple_chat_body("Hello", true);
    let request = chat_request(&body);
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

    let result = collect_stream_result(&events);
    assert!(result.has_role, "first event should have role: assistant");
    assert_eq!(
        result.finish_event_count, 1,
        "should have exactly one finish event"
    );

    let eos_text = "<|endoftext|>";
    assert!(
        result.full_content.ends_with(eos_text),
        "content should end with eos token, got: {:?}",
        result.full_content
    );

    let generated = &result.full_content[..result.full_content.len() - eos_text.len()];
    assert_digit_pattern(generated);
}
