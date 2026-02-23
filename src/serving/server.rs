use async_stream::stream;
use axum::{
    extract::State, response::sse::Event, response::IntoResponse, response::Sse, routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::sync::Notify;

use crate::serving::record::{Phase, SequenceState};
use crate::common::send_sync_ptr::SharedMut;
use crate::serving::batch_sequence::BatchSequence;

// ===== OpenAI API 结构 =====

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StreamResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    index: u32,
    delta: ChatMessage,
    finish_reason: Option<String>,
}

// ===== 应用状态 =====

#[derive(Clone)]
struct AppState {
    batch_sequences: Arc<SharedMut<BatchSequence>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    tokenizer: Arc<Tokenizer>,
}

// ===== HTTP 处理器 =====

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let request_id = format!("chatcmpl-{}", generate_id());
    let is_stream = request.stream.unwrap_or(false);

    // 构造prompt
    let prompt = request
        .messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n");

    // 从全局状态获取并写入，如果没有空位则等待
    let (slot_index, notifier): (usize, Arc<Notify>) = loop {
        let mut found_slot = None;
        {
            let batch_list = unsafe { &mut *state.batch_list.get() };
            let batch_sequences = unsafe { &mut *state.batch_sequences.get() };
            let current_size = batch_list.len();

            for (i, record) in batch_list[..current_size].iter_mut().enumerate() {
                if record.phase == Phase::Eos {
                    match batch_sequences.write_prompt(i, &prompt, &state.tokenizer) {
                        Ok(write_len) => {
                            // record.prompt_length = write_len;
                            record.sequence_index = write_len;
                            record.kv_index = write_len;
                            record.phase = Phase::Decode;
                            found_slot = Some((i, record.notify.clone()));
                            break;
                        }
                        Err(e) => {
                            eprintln!("Error writing prompt: {}", e);
                            return (
                                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Tokenization failed: {}", e),
                            )
                                .into_response();
                        }
                    }
                }
            }
        }

        if let Some(slot) = found_slot {
            break slot;
        }

        // 如果没有空位，等待 1ms 后重试
        tokio::time::sleep(Duration::from_millis(1)).await;
    };

    println!(
        "开始等待推理处理请求: {}, Slot: {}, 模式: {}",
        request_id,
        slot_index,
        if is_stream { "流式" } else { "同步" }
    );

    // 等待推理完成的信号唤醒
    notifier.notified().await;

    // 根据 sequence_index - prompt_length 提取生成的 tokens 并解码
    let generated_text = {
        let (start, end, capacity, sequences_ptr) = {
            let batch_list = unsafe { &*state.batch_list.get() };
            let batch_sequences_meta = unsafe { &*state.batch_sequences.get() };
            let record = &batch_list[slot_index];
            let base = slot_index * batch_sequences_meta.col_size;
            (
                // base + record.prompt_length,
                base + record.sequence_index,
                batch_sequences_meta.row_size * batch_sequences_meta.col_size,
                batch_sequences_meta.sequences,
            )
        };

        if end > start && end <= capacity {
            let mut token_ids: Vec<u32> = Vec::with_capacity(end - start);
            for idx in start..end {
                unsafe {
                    token_ids.push(*sequences_ptr.add(idx) as u32);
                }
            }
            state
                .tokenizer
                .decode(&token_ids, true)
                .unwrap_or_else(|_| String::from("Decode error"))
        } else {
            String::new()
        }
    };

    // 释放槽位并重置状态
    {
        let batch_list = unsafe { &mut *state.batch_list.get() };
        let record = &mut batch_list[slot_index];
        // record.prompt_length = 0;
        record.sequence_index = 0;
        record.kv_index = 0;
        record.phase = Phase::Eos;
    }

    if is_stream {
        // 流式响应
        let model_name = request.model.clone();
        let stream_response = stream! {
            let words: Vec<&str> = generated_text.split_inclusive(' ').collect();
            for (i, word) in words.iter().enumerate() {
                let is_final = i == words.len() - 1;
                let response = StreamResponse {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: model_name.clone(),
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

                tokio::time::sleep(Duration::from_millis(30)).await;
            }
        };

        Sse::new(stream_response).into_response()
    } else {
        // 非流式响应 - 直接执行推理并返回
        println!("同步推理完成: id={}", request_id);

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: request.model,
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
}

async fn status() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "running",
        "mode": "single_threaded_background_processing",
        "info": "Inference and HTTP server run on a single OS thread using current_thread runtime"
    }))
}

fn generate_id() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string()
}

// ===== 主函数 =====

pub async fn run(
    batch_sequences: Arc<SharedMut<BatchSequence>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    tokenizer: Arc<Tokenizer>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("启动单线程架构的 OpenAI 兼容服务器...");

    let state = AppState {
        batch_sequences,
        batch_list,
        tokenizer,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/status", axum::routing::get(status))
        .with_state(state.clone());

    let listener = TcpListener::bind("0.0.0.0:8000").await?;

    println!("服务器运行在 http://0.0.0.0:8000");
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成 (单线程推理模式)");
    println!("  GET  /status - 服务器状态");

    println!("推理由外部 runner 提供，HTTP 仅负责接收请求与响应");

    axum::serve(listener, app).await?;
    Ok(())
}
