use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::runtime::session::SessionMode;
use crate::serving::server::build_router;

use super::test_utils::*;

const NUM_USERS: usize = 20;
const BATCH_SIZE: usize = 32;
const MIN_DELAY_MS: u64 = 50;
const MAX_DELAY_MS: u64 = 200;

#[tokio::test]
async fn test_concurrent_sync_requests() {
    let (manager, _buffer) = create_test_manager_with_mode(BATCH_SIZE, 1000, SessionMode::NonReusable);
    let router = build_router(Arc::clone(&manager));

    let generated_tokens = vec![15496, 995];

    start_generation_worker(Arc::clone(&manager), generated_tokens, NUM_USERS);

    let mut handles = Vec::with_capacity(NUM_USERS);

    for user_id in 0..NUM_USERS {
        let router = router.clone();
        let delay = MIN_DELAY_MS + (user_id as u64 * 7) % (MAX_DELAY_MS - MIN_DELAY_MS);

        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;

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
            assert_eq!(json["choices"][0]["message"]["content"], "Hello world");
            assert_eq!(json["choices"][0]["finish_reason"], "stop");

            (user_id, elapsed)
        });

        handles.push(handle);
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
        "Concurrent test: {} users, avg={:?}, min={:?}, max={:?}",
        NUM_USERS, avg_time, min_time, max_time
    );
}
