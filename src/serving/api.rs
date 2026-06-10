use async_stream::stream;
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Sse},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, Notify, Semaphore};

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch_sequence::BatchSequence;
use crate::runtime::scheduling::{Phase, SequenceState, TokenCounter};

use super::parser::{
    IncrementalStreamingParser, ParserEvent, ParserOptions, StreamingParser, ToolCall,
    ToolCallDelta,
};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct StreamToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: StreamToolFunction,
}

#[derive(Debug, Serialize)]
pub struct StreamToolFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Clone)]
pub struct ApiState {
    pub batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    pub batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    pub token_counter: Arc<TokenCounter>,
    pub parser_options: ParserOptions,
    pub free_slots: Arc<Mutex<VecDeque<usize>>>,
    pub available_slots: Arc<Semaphore>,
}

pub fn build_api_state(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    token_counter: Arc<TokenCounter>,
    parser_options: ParserOptions,
) -> ApiState {
    let initial_free_slots: VecDeque<usize> = batch_states.with(|batch_states_ref| {
        batch_states_ref
            .iter()
            .enumerate()
            .filter_map(|(i, record)| (record.phase == Phase::Start).then_some(i))
            .collect()
    });
    let initial_permits = initial_free_slots.len();

    ApiState {
        batch_sequences,
        batch_states,
        token_counter,
        parser_options,
        free_slots: Arc::new(Mutex::new(initial_free_slots)),
        available_slots: Arc::new(Semaphore::new(initial_permits)),
    }
}

pub(super) async fn chat_completions(
    State(state): State<ApiState>,
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

    let (slot_index, notifier) =
        match assign_slot_with_messages(&state, &request.messages, request.temperature).await {
            Ok(slot) => slot,
            Err(response) => return response,
        };

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        build_stream_response(state, slot_index, notifier, request_id, model, created)
    } else {
        loop {
            notifier.notified().await;

            let is_eos = state.batch_states.with(|batch_list| {
                let record = &batch_list[slot_index];
                matches!(record.phase, Phase::Eos)
            });

            if is_eos {
                break;
            }

            state.token_counter.increment(1).await;
        }

        let generated_text = state.batch_states.with(|batch_list| {
            let record = &batch_list[slot_index];
            state
                .batch_sequences
                .with(|batch_sequences| batch_sequences.decode_generated_text(slot_index, record))
        });
        reclaim_slot(&state, slot_index, true).await;

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

async fn assign_slot_with_messages(
    state: &ApiState,
    messages: &[ChatMessage],
    temperature: Option<f32>,
) -> Result<(usize, Arc<Notify>), axum::response::Response> {
    let permit = state
        .available_slots
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Slot allocator unavailable".to_string(),
            )
                .into_response()
        })?;

    let slot_index = {
        let mut free_slots = state.free_slots.lock().await;
        free_slots.pop_front().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Slot queue empty while permit acquired".to_string(),
            )
                .into_response()
        })?
    };

    let message_pairs = messages
        .iter()
        .map(|msg| (msg.role.as_str(), msg.content.as_str()))
        .collect::<Vec<_>>();

    let (write_len, notifier) = state
        .batch_states
        .with_mut(|batch_list| {
            state.batch_sequences.with_mut(|batch_sequences| {
                let record = &mut batch_list[slot_index];
                if !matches!(record.phase, Phase::Start | Phase::Eos) {
                    Err("slot is not in Start or Eos phase".to_string())
                } else {
                    let temperature = temperature.unwrap_or(1.0);
                    batch_sequences
                        .write_prompts(slot_index, &message_pairs, temperature)
                        .map(|write_len| {
                            record.sequence_index = 0;
                            record.kv_index = 0;
                            record.filling_length = write_len;
                            record.phase = Phase::Prefill;
                            (write_len, record.notify.clone())
                        })
                        .map_err(|e| e.to_string())
                }
            })
        })
        .map_err(|err| {
            eprintln!("Error writing prompt: {}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Tokenization failed: {}", err),
            )
                .into_response()
        })?;

    permit.forget();
    state.token_counter.increment(write_len).await;
    Ok((slot_index, notifier))
}

async fn reclaim_slot(state: &ApiState, slot_index: usize, release_permit: bool) {
    state.batch_states.with_mut(|batch_list| {
        if let Some(record) = batch_list.get_mut(slot_index) {
            record.sequence_index = usize::MAX;
            record.kv_index = usize::MAX;
            record.filling_length = 0;
            record.phase = Phase::Start;
        }
    });

    let mut free_slots = state.free_slots.lock().await;
    free_slots.push_back(slot_index);
    drop(free_slots);

    if release_permit {
        state.available_slots.add_permits(1);
    }
}

fn build_stream_response(
    state: ApiState,
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

            let (token_index, phase) = state.batch_states.with(|batch_list| {
                let record = &batch_list[slot_index];
                (record.sequence_index, record.phase)
            });

            let text = state.batch_sequences.with(|batch_sequences| {
                batch_sequences
                    .decode_single_token(slot_index, token_index)
                    .unwrap_or_default()
            });

            let is_eos = matches!(phase, Phase::Eos);
            if !is_eos {
                state.token_counter.increment(1).await;
            }

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
                    ParserEvent::ToolCallDelta(ToolCallDelta { fragment }) => {
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
                                    arguments: Some(fragment),
                                },
                            }]),
                        };
                        role_sent = true;
                        (delta, None)
                    }
                    ParserEvent::ToolCall(ToolCall { name, arguments }) => {
                        let delta = StreamDelta {
                            role: (!role_sent).then(|| "assistant".to_string()),
                            content: None,
                            reasoning_content: None,
                            tool_calls: Some(vec![StreamToolCall {
                                index: tool_call_index,
                                id: None,
                                kind: "function".to_string(),
                                function: StreamToolFunction {
                                    name: Some(name),
                                    arguments: Some(arguments.to_string()),
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

        reclaim_slot(&state, slot_index, true).await;
    };

    Sse::new(stream_body).into_response()
}
