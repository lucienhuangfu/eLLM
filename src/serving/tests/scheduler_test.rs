use axum::http::StatusCode;
use std::sync::Arc;
use tower::ServiceExt;

use crate::runtime::session::SessionMode;

use super::test_utils::*;

const EOS_TEXT_QWEN3: &str = "<|im_end|>";

fn setup_qwen3_fakeecho(
    batch_size: usize,
    mode: SessionMode,
    threads: usize,
    max_gen: usize,
) -> (axum::Router, Arc<crate::runtime::session::SlotManager<f16>>) {
    let (manager, _buffer) = create_qwen3_test_manager_with_mode(batch_size, 5000, mode);
    let router = crate::serving::server::build_router(Arc::clone(&manager));
    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let gen_tokens: Vec<usize> = (0..max_gen).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, threads, gen_tokens);
    (router, manager)
}

// ─── Basic sync FakeEcho (qwen3 tokenizer) ─────────────────────────────────

#[tokio::test]
async fn test_sync_fakeecho_qwen3() {
    let (router, _manager) = setup_qwen3_fakeecho(4, SessionMode::NonReusable, 1, 10);

    let body = simple_chat_body("Hello", false);
    let request = chat_request(&body);
    let response = router.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let resp_body = collect_body(response.into_body()).await;
    let json: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();

    assert_sync_response_basic(&json);
    let generated = assert_sync_content_ends_with(&json, EOS_TEXT_QWEN3);
    assert_digit_pattern(&generated);
}
