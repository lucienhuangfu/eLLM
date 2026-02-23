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

use crate::serving::record::Phase;

use super::AppState;
use super::types::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, StreamChoice,
    StreamResponse,
};

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
    let prompt = build_prompt(&request.messages);

    let (slot_index, notifier) = match assign_slot_with_prompt(&state, &prompt).await {
        Ok(slot) => slot,
        Err(response) => return response,
    };

    println!(
        "开始等待推理处理请求: {}, Slot: {}, 模式: {}",
        request_id,
        slot_index,
        if is_stream { "流式" } else { "同步" }
    );

    wait_for_inference(&notifier).await;

    let generated_text = decode_generated_text(&state, slot_index);
    recycle_slot(&state, slot_index).await;

    if is_stream {
        build_stream_response(request_id, request.model, generated_text)
    } else {
        println!("同步推理完成: id={}", request_id);
        build_non_stream_response(request_id, request.model, generated_text)
    }
}

async fn assign_slot_with_prompt(
    state: &AppState,
    prompt: &str,
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

    let assign_result = {
        let batch_list = unsafe { &mut *state.batch_list.get() };
        let batch_sequences = unsafe { &mut *state.batch_sequences.get() };

        let record = &mut batch_list[slot_index];
        if !matches!(record.phase, Phase::Start) {
            Err("slot is not in Start phase".to_string())
        } else {
            batch_sequences
                .write_prompt(slot_index, prompt, &state.tokenizer)
                .map(|write_len| {
                    record.sequence_index = write_len;
                    record.kv_index = 0;
                    record.phase = Phase::Prefill;
                    record.notify.clone()
                })
                .map_err(|e| e.to_string())
        }
    };

    match assign_result {
        Ok(notifier) => {
            permit.forget();
            Ok((slot_index, notifier))
        }
        Err(e) => {
            {
                let batch_list = unsafe { &mut *state.batch_list.get() };
                let record = &mut batch_list[slot_index];
                record.sequence_index = 0;
                record.kv_index = 0;
                record.phase = Phase::Start;
            }
            let mut free_slots = state.free_slots.lock().await;
            free_slots.push_back(slot_index);
            drop(free_slots);
            drop(permit);

            eprintln!("Error writing prompt: {}", e);
            Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Tokenization failed: {}", e),
            )
                .into_response())
        }
    }
}

async fn wait_for_inference(notifier: &Arc<Notify>) {
    notifier.notified().await;
}

async fn recycle_slot(state: &AppState, slot_index: usize) {
    {
        let batch_list = unsafe { &mut *state.batch_list.get() };
        if let Some(record) = batch_list.get_mut(slot_index) {
            record.sequence_index = 0;
            record.kv_index = 0;
            record.phase = Phase::Start;
        }
    }

    let mut free_slots = state.free_slots.lock().await;
    free_slots.push_back(slot_index);
    drop(free_slots);
    state.available_slots.add_permits(1);
}

fn decode_generated_text(state: &AppState, slot_index: usize) -> String {
    let (start, end, capacity, sequences_ptr) = {
        let batch_list = unsafe { &*state.batch_list.get() };
        let batch_sequences_meta = unsafe { &*state.batch_sequences.get() };
        let record = &batch_list[slot_index];
        let base = slot_index * batch_sequences_meta.col_size;
        (
            base + record.sequence_index,
            base + record.kv_index,
            batch_sequences_meta.row_size * batch_sequences_meta.col_size,
            batch_sequences_meta.sequences,
        )
    };

    if end <= start || end > capacity {
        return String::new();
    }

    let token_ids: Vec<u32> = unsafe {
        let token_slice = std::slice::from_raw_parts(sequences_ptr.add(start), end - start);
        token_slice.iter().map(|&id| id as u32).collect()
    };

    state
        .tokenizer
        .decode(&token_ids, true)
        .unwrap_or_else(|_| String::from("Decode error"))
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

fn build_prompt(messages: &[ChatMessage]) -> String {
    let estimated_len: usize = messages
        .iter()
        .map(|msg| msg.role.len() + msg.content.len() + 3)
        .sum();
    let mut prompt = String::with_capacity(estimated_len);

    for (i, msg) in messages.iter().enumerate() {
        if i > 0 {
            prompt.push('\n');
        }
        prompt.push_str(&msg.role);
        prompt.push_str(": ");
        prompt.push_str(&msg.content);
    }

    prompt
}
