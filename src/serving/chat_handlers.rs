use async_stream::stream;
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Sse},
    Json,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Notify;

use crate::runtime::Phase;

use super::{
    ApiState, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    StreamChoice, StreamResponse,
};

fn build_error_response(code: StatusCode, message: &str) -> axum::response::Response {
    (code, message.to_string()).into_response()
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

    let (slot_index, notifier, prompt_len) =
        match assign_slot_with_messages(&state, &request.messages, request.temperature).await {
            Ok(slot) => slot,
            Err(response) => return response,
        };

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        build_stream_response(
            state, slot_index, notifier, prompt_len, request_id, model, created,
        )
    } else {
        // Non-streaming: wait for the single EOS notification, then decode all
        // generated tokens at once.
        notifier.notified().await;

        let generated_text = state.batch_states.with(|batch_list| {
            let record = &batch_list[slot_index];
            state
                .batch_sequences
                .with(|batch_sequences| batch_sequences.decode_generated_text(slot_index, record))
        });
        reclaim_slot(&state, slot_index, true).await;

        #[cfg(debug_assertions)]
        println!("同步推理完成: id={}", request_id);

        build_non_stream_response(request_id, model, created, generated_text)
    }
}

/// Returns `(slot_index, notifier, prompt_len)`.
async fn assign_slot_with_messages(
    state: &ApiState,
    messages: &[ChatMessage],
    temperature: Option<f32>,
) -> Result<(usize, Arc<Notify>, usize), axum::response::Response> {
    let permit = state
        .available_slots
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| {
            build_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Slot allocator unavailable",
            )
        })?;

    let slot_index = {
        let mut free_slots = state.free_slots.lock().await;
        free_slots.pop_front().ok_or_else(|| {
            build_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Slot queue empty while permit acquired",
            )
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
            build_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Tokenization failed: {}", err),
            )
        })?;

    permit.forget();
    state.token_counter.increment(write_len).await;
    Ok((slot_index, notifier, write_len))
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

/// True incremental streaming: each `notify_one()` from `TopKSoftmax`
/// corresponds to one decoded token. We read `record.sequence_index` (the
/// position just written) after every wake-up, decode that single token, and
/// push it as an SSE chunk immediately. When `phase == Eos` we emit the final
/// chunk with `finish_reason: "stop"` and close the stream.
fn build_stream_response(
    state: ApiState,
    slot_index: usize,
    notifier: Arc<Notify>,
    _prompt_len: usize,
    request_id: String,
    model: String,
    created: u64,
) -> axum::response::Response {
    let stream_body = stream! {
        loop {
            notifier.notified().await;

            // Read the token position and phase written by TopKSoftmax.
            let (token_index, phase) = state.batch_states.with(|batch_list| {
                let record = &batch_list[slot_index];
                (record.sequence_index, record.phase)
            });

            // Decode the single token at token_index.
            let text = state.batch_sequences.with(|batch_sequences| {
                batch_sequences
                    .decode_single_token(slot_index, token_index)
                    .unwrap_or_default()
            });

            let is_eos = matches!(phase, Phase::Eos);

            let response = StreamResponse {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: ChatMessage {
                        role: "assistant".to_string(),
                        content: text,
                    },
                    finish_reason: is_eos.then(|| "stop".to_string()),
                }],
            };

            if let Ok(json) = serde_json::to_string(&response) {
                yield Ok::<Event, axum::Error>(Event::default().data(json));
            }

            if is_eos {
                break;
            }
        }

        reclaim_slot(&state, slot_index, true).await;
    };

    Sse::new(stream_body).into_response()
}

fn build_non_stream_response(
    request_id: String,
    model: String,
    created: u64,
    generated_text: String,
) -> axum::response::Response {
    let response = ChatCompletionResponse {
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
    };

    Json(response).into_response()
}
