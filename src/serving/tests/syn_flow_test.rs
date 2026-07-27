use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;

use super::test_utils::*;

#[tokio::test]
async fn test_chat_completions_sync_full_flow() {
    let (router, manager, _buffer) = create_test_router();

    start_generation_loop(Arc::clone(&manager), vec![15496, 995]);

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
    assert_eq!(json["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}
