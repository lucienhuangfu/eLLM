use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;

use crate::runtime::session::SessionMode;
use crate::serving::server::build_router;

use super::test_utils::*;

#[tokio::test]
async fn test_chat_completions_stream_full_flow() {
    let (manager, _buffer) = create_test_manager_with_mode(4, 1000, SessionMode::NonReusable);
    let router = build_router(Arc::clone(&manager));

    let digit_tokens = digit_tokens_r50k();
    let eos_id = 50256;
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
    assert!(body_str.contains("0"));
    assert!(body_str.contains("finish_reason"));
}
