use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;

use crate::operators::fake_echo::FakeEcho;
use crate::operators::operator::Operator;
use crate::runtime::executor::executor_pool::ExecutorPool;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{SessionMode, SlotManager};

use super::test_utils::*;

#[tokio::test]
async fn test_chat_completions_with_scheduler_fakeecho() {
    let (router, manager, _buffer) = create_test_router_with_mode(SessionMode::NonReusable);

    let eos_id = 50256;
    let _scheduler = start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, 1);

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
}

fn start_runtime_with_fakeecho(
    manager: Arc<SlotManager<f16>>,
    eos_id: usize,
    thread_num: usize,
) -> Arc<Scheduler> {
    let (batch_size, seq_len, sequences_ptr) = manager.batch_sequences.with(|seq| {
        (seq.row_size, seq.col_size, seq.sequences)
    });

    let batch_states = manager.batch_states.clone();
    let scheduler = Arc::new(Scheduler::new(batch_size, seq_len, thread_num, batch_states));

    let fake_echo = FakeEcho::new(sequences_ptr, seq_len, eos_id);
    let operator_queue: Vec<Operator<f16>> = vec![Operator::FakeEcho(fake_echo)];

    let executor = ExecutorPool::new(operator_queue, Arc::clone(&scheduler), thread_num);
    executor.start();

    scheduler
}
