mod test_utils;
mod syn_flow_test;
mod stream_flow_test;
mod concurrent_test;
mod scheduler_fakeecho_test;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use test_utils::*;

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

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}
