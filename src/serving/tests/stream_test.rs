use axum::http::StatusCode;
use std::sync::Arc;
use tower::ServiceExt;

use crate::runtime::session::SessionMode;

use super::test_utils::*;

const EOS_TEXT_QWEN3: &str = "<|im_end|>";

fn setup_stream_fakeecho(
    batch_size: usize,
    reasoning: bool,
    tool_call: bool,
    threads: usize,
) -> (axum::Router, Arc<crate::runtime::session::SlotManager<f16>>) {
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        batch_size,
        5000,
        SessionMode::NonReusable,
        reasoning,
        tool_call,
    );
    let router = crate::serving::server::build_router(Arc::clone(&manager));
    let digit_tokens: Vec<usize> = (15..25).collect();
    let eos_id = 151645;
    let gen_tokens: Vec<usize> = (0..10).map(|i| digit_tokens[i % digit_tokens.len()]).collect();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, threads, gen_tokens);
    (router, manager)
}

fn setup_stream_fake_parser_echo(
    batch_size: usize,
    threads: usize,
    script_text: &str,
) -> (axum::Router, Arc<crate::runtime::session::SlotManager<f16>>) {
    let (manager, _buffer) = create_qwen3_test_manager_with_parser(
        batch_size,
        5000,
        SessionMode::NonReusable,
        true,
        true,
    );
    let router = crate::serving::server::build_router(Arc::clone(&manager));
    let tokenizer = qwen3_tokenizer();
    let eos_id = 151645;
    let script_tokens = tokenizer
        .encode_with_special_tokens(script_text)
        .into_iter()
        .map(|t| t as usize)
        .collect::<Vec<_>>();
    let _scheduler =
        start_runtime_with_fakeecho(Arc::clone(&manager), eos_id, threads, script_tokens);
    (router, manager)
}

// ─── Stream FakeEcho (parser=false) ────────────────────────────────────────

#[tokio::test]
async fn test_stream_fakeecho_parser_false() {
    let (router, _manager) = setup_stream_fakeecho(4, false, false, 1);

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

    for event in &events {
        assert_sse_event_basics(event);
    }

    let result = collect_stream_result(&events);
    assert!(result.has_role, "first event should have role: assistant");
    assert_eq!(
        result.finish_event_count, 1,
        "should have exactly one finish event"
    );
    assert!(
        !result.full_content.is_empty(),
        "generated content should not be empty"
    );
    assert!(
        result.full_content.ends_with(EOS_TEXT_QWEN3),
        "content should end with eos token, got: {:?}",
        result.full_content
    );

    let generated = &result.full_content[..result.full_content.len() - EOS_TEXT_QWEN3.len()];
    assert_digit_pattern(generated);
}

// ─── Stream FakeParserEcho (parser=true) ───────────────────────────────────

#[tokio::test]
async fn test_stream_fake_parser_echo_parser_true() {
    let script_text = concat!(
        "Hello, let me think about this.",
        "<think>Analyzing the user request step by step.</think>",
        "I need to call a tool for this.",
        "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"weather in Beijing\"}}</tool_call>",
        "The tool returned the result."
    );

    let expected_content = format!(
        "{}{}{}{}",
        "Hello, let me think about this.",
        "I need to call a tool for this.",
        "The tool returned the result.",
        EOS_TEXT_QWEN3
    );
    let expected_reasoning = "Analyzing the user request step by step.";
    let expected_tool_name = "search";
    let expected_tool_args = r#"{"query":"weather in Beijing"}"#;

    let (router, _manager) = setup_stream_fake_parser_echo(4, 1, script_text);

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

    for event in &events {
        assert_sse_event_basics(event);
    }

    let result = collect_stream_result(&events);
    assert!(result.has_role, "first event should have role: assistant");
    assert_eq!(
        result.finish_event_count, 1,
        "should have exactly one finish event"
    );

    assert!(
        result.has_reasoning,
        "should have reasoning_content events when reasoning_parser is true"
    );
    assert!(
        result.has_tool_calls,
        "should have tool_calls events when tool_call_parser is true"
    );

    assert_eq!(
        result.full_content, expected_content,
        "full content mismatch.\nexpected: {:?}\ngot: {:?}",
        expected_content, result.full_content
    );

    assert_eq!(
        result.full_reasoning, expected_reasoning,
        "full reasoning mismatch.\nexpected: {:?}\ngot: {:?}",
        expected_reasoning, result.full_reasoning
    );

    assert_eq!(
        result.tool_calls.len(),
        1,
        "should have exactly 1 tool call, got {}",
        result.tool_calls.len()
    );

    assert_eq!(
        result.tool_calls[0].name, expected_tool_name,
        "tool call name mismatch.\nexpected: {:?}\ngot: {:?}",
        expected_tool_name, result.tool_calls[0].name
    );

    assert_eq!(
        result.tool_calls[0].arguments, expected_tool_args,
        "tool call arguments mismatch.\nexpected: {:?}\ngot: {:?}",
        expected_tool_args, result.tool_calls[0].arguments
    );
}
