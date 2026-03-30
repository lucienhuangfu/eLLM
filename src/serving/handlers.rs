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

use crate::runtime::Phase;

use super::types::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, StreamChoice,
    StreamResponse,
};
use super::bootstrap::AppState;

pub(super) async fn chat_completions(
    State(state): State<AppState>,
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

    let (slot_index, notifier) = match assign_slot_with_messages(&state, &request.messages).await {
        Ok(slot) => slot,
        Err(response) => return response,
    };

    println!(
        "开始等待推理处理请求: {}, Slot: {}, 模式: {}",
        request_id,
        slot_index,
        if is_stream { "流式" } else { "同步" }
    );

    notifier.notified().await;

    let generated_text = state.batch_list.with(|batch_list| {
        state.batch_sequences.with(|batch_sequences| {
            let record = &batch_list[slot_index];
            batch_sequences.decode_generated_text(slot_index, record)
        })
    });
    reclaim_slot(&state, slot_index, true).await;

    if is_stream {
        build_stream_response(request_id, request.model, generated_text)
    } else {
        println!("同步推理完成: id={}", request_id);
        build_non_stream_response(request_id, request.model, generated_text)
    }
}

async fn assign_slot_with_messages(
    state: &AppState,
    messages: &[ChatMessage],
) -> Result<(usize, Arc<Notify>), axum::response::Response> {
    let permit = match state.available_slots.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            return Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "Slot allocator unavailable",
            )
                .into_response());
        }
    };

    let slot_index = {
        let mut free_slots = state.free_slots.lock().await;
        match free_slots.pop_front() {
            Some(index) => index,
            None => {
                return Err((
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Slot queue empty while permit acquired",
                )
                    .into_response());
            }
        }
    };

    let message_pairs = messages
        .iter()
        .map(|msg| (msg.role.as_str(), msg.content.as_str()))
        .collect::<Vec<_>>();

    let write_result = state.batch_list.with_mut(|batch_list| {
        state.batch_sequences.with_mut(|batch_sequences| {
            let record = &mut batch_list[slot_index];
            if !matches!(record.phase, Phase::Start) {
                Err("slot is not in Start phase".to_string())
            } else {
                batch_sequences
                    .write_prompts(slot_index, &message_pairs)
                    .map(|write_len| {
                        record.sequence_index = 0;
                        record.kv_index = 0;
                        record.filling_length = write_len;
                        record.phase = Phase::Prefill;
                        record.notify.clone()
                    })
                    .map_err(|e| e.to_string())
            }
        })
    });

    match write_result {
        Ok(notifier) => {
            permit.forget();
            Ok((slot_index, notifier))
        }
        Err(err) => {
            reclaim_slot(state, slot_index, false).await;
            drop(permit);

            eprintln!("Error writing prompt: {}", err);
            Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Tokenization failed: {}", err),
            )
                .into_response())
        }
    }
}

async fn reclaim_slot(state: &AppState, slot_index: usize, release_permit: bool) {
    state.batch_list.with_mut(|batch_list| {
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
    request_id: String,
    model: String,
    generated_text: String,
) -> axum::response::Response {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let stream_response = stream! {
        let mut words = generated_text.split_inclusive(' ').peekable();
        while let Some(word) = words.next() {
            let is_final = words.peek().is_none();
            let response = StreamResponse {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: ChatMessage {
                        role: "assistant".to_string(),
                        content: word.to_string(),
                    },
                    finish_reason: if is_final { Some("stop".to_string()) } else { None },
                }],
            };

            if let Ok(json) = serde_json::to_string(&response) {
                yield Ok::<Event, axum::Error>(Event::default().data(json));
            }
        }
    };

    Sse::new(stream_response).into_response()
}

fn build_non_stream_response(
    request_id: String,
    model: String,
    generated_text: String,
) -> axum::response::Response {
    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
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
