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
use tokio::sync::{Notify, RwLock};

use crate::init::record::{BatchRecord, Phase};

// ===== OpenAI API 结构 =====

#[derive(Debug)]
struct BatchPrompt {
    tokens: Vec<u32>, // 展平的二维矩阵 [batch_size][capacity]
    batch_size: usize,
    capacity: usize,
}

impl BatchPrompt {
    fn new(batch_size: usize, capacity: usize) -> Self {
        Self {
            tokens: vec![0; batch_size * capacity],
            batch_size,
            capacity,
        }
    }

    fn write_prompt(
        &mut self,
        slot_index: usize,
        prompt: &str,
        tokenizer: &Tokenizer,
    ) -> Result<usize, String> {
        // 使用真正的分词器进行编码
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        let ids = tokens.get_ids();
        let write_len = ids.len().min(self.capacity);

        // 计算在展平矩阵中的起始偏移量
        let offset = slot_index * self.capacity;

        // 将 tokens 写入对应 slot 的 buffer
        self.tokens[offset..offset + write_len].copy_from_slice(&ids[..write_len]);

        println!(
            "Prompt 已通过 Tokenizer 写入 BatchPrompt Slot {}, 长度: {}",
            slot_index, write_len
        );
        Ok(write_len)
    }
}

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
    batch_prompts: Arc<RwLock<BatchPrompt>>,
    batch_list: Arc<RwLock<Vec<BatchRecord>>>,
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
            let mut batch_list = state.batch_list.write().await;
            let mut batch_prompts = state.batch_prompts.write().await;
            let current_size = batch_list.len();

            for (i, record) in batch_list.iter_mut().take(current_size).enumerate() {
                if record.prompt_length == 0 {
                    match batch_prompts.write_prompt(i, &prompt, &state.tokenizer) {
                        Ok(write_len) => {
                            record.prompt_length = write_len;
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
        let (start, end, capacity) = {
            let batch_list = state.batch_list.read().await;
            let record = &batch_list[slot_index];
            let batch_prompts_meta = state.batch_prompts.read().await;
            let base = slot_index * batch_prompts_meta.capacity;
            (
                base + record.prompt_length,
                base + record.sequence_index,
                batch_prompts_meta.tokens.len(),
            )
        };

        if end > start && end <= capacity {
            let batch_prompts = state.batch_prompts.read().await;
            state
                .tokenizer
                .decode(&batch_prompts.tokens[start..end], true)
                .unwrap_or_else(|_| String::from("Decode error"))
        } else {
            String::new()
        }
    };

    // 释放槽位并重置状态
    {
        let mut batch_list = state.batch_list.write().await;
        let record = &mut batch_list[slot_index];
        record.prompt_length = 0;
        record.sequence_index = 0;
        record.kv_index = 0;
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

#[tokio::main(flavor = "current_thread")]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("启动单线程架构的 OpenAI 兼容服务器...");

    // 加载分词器
    let tokenizer_path = "models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("无法加载分词器 {}: {}", tokenizer_path, e))?;
    let tokenizer = Arc::new(tokenizer);

    // 在 main 中创建全局共享的 batch_prompts 和 batch_list
    let batch_size = 4;
    let capacity = 50000;
    let batch_prompts = Arc::new(RwLock::new(BatchPrompt::new(batch_size, capacity)));

    let batch_records = (0..batch_size)
        .map(|_| BatchRecord {
            sequence_index: 0,
            snapshot_sequence_index: 0,
            kv_index: 0,
            phase: Phase::PrefillBegin,
            prompt_length: 0,
            notify: Arc::new(tokio::sync::Notify::new()),
        })
        .collect::<Vec<_>>();

    let batch_list = Arc::new(RwLock::new(batch_records));

    let state = AppState {
        batch_prompts,
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

    // 启动背景推理循环 (在同一个单线程 runtime 上运行)
    let loop_state = state.clone();
    tokio::spawn(async move {
        println!("背景推理任务已启动");
        loop {
            let mut processed = false;
            {
                let mut batch_list = loop_state.batch_list.write().await;
                let current_size = batch_list.len();

                for (i, record) in batch_list.iter_mut().take(current_size).enumerate() {
                    // 如果有 prompt 且处于等待推理的状态
                    if record.prompt_length > 0 && record.sequence_index == 0 {
                        // 模拟推理逻辑：
                        // 在现实中，这里会启动真正的推理引擎。

                        // 模拟生成过程：写入几个简单的 token id
                        {
                            let mut batch_prompts = loop_state.batch_prompts.write().await;
                            let start = i * batch_prompts.capacity + record.prompt_length;
                            // 模拟写入 "Hello" 类似的 token ids
                            for j in 0..5 {
                                if start + j < batch_prompts.tokens.len() {
                                    batch_prompts.tokens[start + j] = 77 + (j as u32);
                                }
                            }
                        }

                        // 设定生成的长度
                        record.sequence_index = record.prompt_length + 5;

                        record.notify.notify_one();
                        println!("Slot {} 推理完成 (单线程模拟)", i);
                        processed = true;
                    }
                }
            }

            if !processed {
                // 如果没有待处理任务，稍微休眠以避免 100% CPU 占用
                tokio::time::sleep(Duration::from_millis(5)).await;
            } else {
                // 处理完一批后，yield 让出执行权给 HTTP server
                tokio::task::yield_now().await;
            }
        }
    });

    axum::serve(listener, app).await?;
    Ok(())
}
