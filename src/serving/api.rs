use async_stream::stream;
use axum::{
    extract::State,
    response::sse::Event,
    response::{IntoResponse, Sse},
    Json,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Notify;

use super::parser::{IncrementalStreamingParser, ParserEvent, StreamingParser};
use super::requests::{ChatCompletionRequest, ChatMessage};
use super::responses::{ChatCompletionChoice, ChatCompletionResponse};
use super::state::ApiState;
use super::stream::{
    StreamChoice, StreamDelta, StreamResponse, StreamToolCall, StreamToolFunction,
};

pub(super) async fn chat_completions(
    State(state): State<ApiState<f16>>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let request_id = format!(
        "chatcmpl-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let is_stream = request.stream.unwrap_or(false);
    let model = request.model;

    let slot_index = match state.acquire_slot().await {
        Ok(slot) => slot,
        Err(response) => return response,
    };

    let (write_len, notifier) =
        match state.write_prompts_and_prepare(slot_index, &request.messages, request.temperature) {
            Ok(result) => result,
            Err(response) => {
                state.release_slot(slot_index, true).await;
                return response;
            }
        };

    state.scheduler.notify_tokens(write_len).await;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        build_stream_response(state, slot_index, notifier, request_id, model, created)
    } else {
        loop {
            notifier.notified().await;

            if state.is_eos(slot_index) {
                break;
            }
        }

        let generated_text = state.decode_generated_text(slot_index);
        state.release_slot(slot_index, true).await;

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
    state: ApiState<f16>,
    slot_index: usize,
    notifier: Arc<Notify>,
    request_id: String,
    model: String,
    created: u64,
) -> axum::response::Response {
    let mut parser = IncrementalStreamingParser::with_options(state.parser_options);
    let mut role_sent = false;
    let mut tool_call_index = 0u32;
    let stream_body = stream! {
        loop {
            notifier.notified().await;

            let (token_index, phase) = state.get_token_index_and_phase(slot_index);
            let text = state.decode_single_token(slot_index, token_index);
            let is_eos = matches!(phase, crate::runtime::scheduling::Phase::Eos);

            let mut events = parser.feed(&text);
            if is_eos {
                events.push(ParserEvent::Finish);
            }

            for event in events {
                let (delta, finish_reason) = match event {
                    ParserEvent::Content(content) => {
                        let delta = StreamDelta {
                            role: (!role_sent).then(|| "assistant".to_string()),
                            content: Some(content),
                            reasoning_content: None,
                            tool_calls: None,
                        };
                        role_sent = true;
                        (delta, None)
                    }
                    ParserEvent::Reasoning(reasoning) => {
                        let delta = StreamDelta {
                            role: (!role_sent).then(|| "assistant".to_string()),
                            content: None,
                            reasoning_content: Some(reasoning),
                            tool_calls: None,
                        };
                        role_sent = true;
                        (delta, None)
                    }
                    ParserEvent::ToolCallDelta(delta) => {
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
                    ParserEvent::ToolCall(tool_call) => {
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
                    ParserEvent::Finish => (StreamDelta::default(), Some("stop".to_string())),
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

        state.release_slot(slot_index, true).await;
    };

    Sse::new(stream_body).into_response()
}
