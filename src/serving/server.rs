use async_stream::stream;
use axum::{
    extract::State,
    response::sse::Event,
    response::{IntoResponse, Sse},
    routing::post,
    Json, Router,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::Notify;

use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::SlotManager;
use crate::runtime::Backend;
use super::parser::{IncrementalStreamingParser, ParserOptions, StreamingParser};
use super::types::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    StreamChoice, StreamDelta, StreamResponse, StreamToolCall, StreamToolFunction,
};

// ── Route handlers ──────────────────────────────────────────

async fn chat_completions(
    State(state): State<Backend>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let request_id = request
        .request_id
        .unwrap_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4()));
    let is_stream = request.stream.unwrap_or(false);
    let model = request.model;

    let session_id = request.session_id.unwrap_or_else(|| request_id.clone());

    let handle = match state.acquire_session(&session_id).await {
        Ok(h) => h,
        Err(e) => return e.into_response(),
    };

    let slot_index = handle.slot_index;

    let (write_len, notifier) = match if handle.is_reused {
        state
            .write_prompts_with_incremental_prefill(
                slot_index,
                &session_id,
                &request.messages,
                request.temperature,
            )
            .await
    } else {
        state
            .write_prompts_and_prepare(slot_index, &request.messages, request.temperature)
            .await
    } {
        Ok(result) => result,
        Err(e) => {
            state.release_session(&session_id, 0).await;
            return e.into_response();
        }
    };

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        build_stream_response(
            state,
            slot_index,
            &session_id,
            notifier,
            request_id,
            model,
            created,
        )
    } else {
        while !state.is_eos(slot_index) {
            notifier.notified().await;
        }

        let generated_text = state.decode_generated_text(slot_index);
        let token_count = state.get_sequence_index(slot_index);
        state.release_session(&session_id, token_count).await;

        #[cfg(debug_assertions)]
        println!("同步推理完成: id={}", request_id);

        Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: generated_text,
                },
                finish_reason: Some("stop".to_string()),
            }],
        })
        .into_response()
    }
}

fn build_stream_response(
    state: Backend,
    slot_index: usize,
    session_id: &str,
    notifier: Arc<Notify>,
    request_id: String,
    model: String,
    created: u64,
) -> axum::response::Response {
    let session_id = session_id.to_string();
    let mut parser = IncrementalStreamingParser::with_options(state.parser_options);
    let mut role_sent = false;
    let mut tool_call_index = 0u32;

    let stream_body = stream! {
        loop {
            notifier.notified().await;

            let (token_index, phase) = state.get_token_index_and_phase(slot_index);
            let text = state.decode_single_token(slot_index, token_index);
            let is_eos = matches!(phase, crate::runtime::Phase::Eos);

            let mut events = parser.feed(&text);
            if is_eos {
                events.push(super::parser::ParserEvent::Finish);
            }

            for event in events {
                let (delta, finish_reason) = match event {
                    super::parser::ParserEvent::Content(content) => {
                        let delta = StreamDelta {
                            role: (!role_sent).then(|| "assistant".to_string()),
                            content: Some(content),
                            reasoning_content: None,
                            tool_calls: None,
                        };
                        role_sent = true;
                        (delta, None)
                    }
                    super::parser::ParserEvent::Reasoning(reasoning) => {
                        let delta = StreamDelta {
                            role: (!role_sent).then(|| "assistant".to_string()),
                            content: None,
                            reasoning_content: Some(reasoning),
                            tool_calls: None,
                        };
                        role_sent = true;
                        (delta, None)
                    }
                    super::parser::ParserEvent::ToolCallDelta(delta) => {
                        let delta = StreamDelta {
                            role: (!role_sent).then(|| "assistant".to_string()),
                            content: None,
                            reasoning_content: None,
                            tool_calls: Some(vec![StreamToolCall {
                                index: tool_call_index,
                                id: None,
                                kind: "function".to_string(),
                                function: StreamToolFunction {
                                    name: None,
                                    arguments: Some(delta.fragment),
                                },
                            }]),
                        };
                        role_sent = true;
                        (delta, None)
                    }
                    super::parser::ParserEvent::ToolCall(tool_call) => {
                        let delta = StreamDelta {
                            role: (!role_sent).then(|| "assistant".to_string()),
                            content: None,
                            reasoning_content: None,
                            tool_calls: Some(vec![StreamToolCall {
                                index: tool_call_index,
                                id: None,
                                kind: "function".to_string(),
                                function: StreamToolFunction {
                                    name: Some(tool_call.name),
                                    arguments: Some(tool_call.arguments.to_string()),
                                },
                            }]),
                        };
                        tool_call_index += 1;
                        role_sent = true;
                        (delta, None)
                    }
                    super::parser::ParserEvent::Finish => (StreamDelta::default(), Some("stop".to_string())),
                };

                let response = StreamResponse {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta,
                        finish_reason,
                    }],
                };

                if let Ok(json) = serde_json::to_string(&response) {
                    yield Ok::<Event, axum::Error>(Event::default().data(json));
                }
            }

            if is_eos {
                break;
            }
        }

        let token_count = state.get_sequence_index(slot_index);
        state.release_session(&session_id, token_count).await;
    };

    Sse::new(stream_body).into_response()
}

pub(crate) fn build_router(state: Backend) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route(
            "/status",
            axum::routing::get(|| async {
                Json(serde_json::json!({
                    "status": "running",
                    "mode": "inlined_scheduler",
                    "info": "Scheduler is inlined in worker loop, executed by leader thread"
                }))
            }),
        )
        .with_state(state)
}

pub async fn run(
    scheduler: Arc<Scheduler>,
    parser_options: ParserOptions,
    slot_manager: Arc<SlotManager<f16>>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("启动事件驱动的 OpenAI 兼容服务器...");

    let state = Backend::new(scheduler, slot_manager, parser_options);

    let app = build_router(state);

    let listener = TcpListener::bind("0.0.0.0:8000").await?;

    println!("服务器运行在 http://0.0.0.0:8000");
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成");
    println!("  GET  /status - 服务器状态");
    println!("调度由 leader worker 线程内联执行");

    axum::serve(listener, app).await?;
    Ok(())
}
