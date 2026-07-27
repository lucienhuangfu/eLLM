use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;

use crate::runtime::session::SessionMode;
use crate::serving::server::build_router;

use super::test_utils::*;

#[tokio::test]
async fn test_chat_completions_sync_full_flow() {
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

    let eos_text = "<|endoftext|>";
    assert!(
        content.ends_with(eos_text),
        "content should end with eos token <|endoftext|>, got: {:?}",
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
