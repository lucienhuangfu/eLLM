use crate::init::record::{BatchList, BatchRecord, Phase};
use async_stream::stream;
use axum::{
    extract::State, response::sse::Event, response::IntoResponse, response::Sse, routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;

// ===== OpenAI API 结构 =====

#[derive(Debug)]
struct BatchPrompt {
    tokens: Vec<i32>,
    len: usize,
}

impl BatchPrompt {
    fn new(capacity: usize) -> Self {
        Self {
            tokens: vec![0; capacity],
            len: 0,
        }
    }

    fn write_prompt(&mut self, prompt: &str) {
        // 模拟分词过程，实际应调用 tokenizer.encode
        let mock_tokens: Vec<i32> = prompt.as_bytes().iter().map(|&b| b as i32).collect();
        let write_len = mock_tokens.len().min(self.tokens.len());

        // 将 prompt 一次性写入预分配的 buffer
        self.tokens[..write_len].copy_from_slice(&mock_tokens[..write_len]);
        self.len = write_len;

        println!("Prompt 已写入 BatchPrompt, 长度: {}", self.len);
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
    batch_prompts: Arc<Mutex<Vec<BatchPrompt>>>,
    batch_list: Arc<Mutex<BatchList>>,
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
    let (slot_index, notifier) = loop {
        let mut found_slot = None;
        {
            let mut batch_list = state.batch_list.lock().unwrap();
            let mut batch_prompts = state.batch_prompts.lock().unwrap();

            for (i, record) in batch_list.records[..batch_list.current_size]
                .iter_mut()
                .enumerate()
            {
                if record.sequence_length == 0 {
                    batch_prompts[i].write_prompt(&prompt);
                    record.sequence_length = batch_prompts[i].len;
                    found_slot = Some((i, record.notify.clone()));
                    break;
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

    if is_stream {
        // 流式响应
        let stream_response = stream! {
            // 模拟推理开始前的延迟
            tokio::time::sleep(Duration::from_millis(100)).await;

            let result = format!(
                "This is a direct response to '{}' using model {}. Inference is now nested inside the handler without background workers.",
                prompt.replace("\n", " "), request.model
            );

            let words: Vec<&str> = result.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                let is_final = i == words.len() - 1;
                let response = StreamResponse {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: request.model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: ChatMessage {
                            role: "assistant".to_string(),
                            content: format!("{} ", word),
                        },
                        finish_reason: if is_final { Some("stop".to_string()) } else { None },
                    }],
                };

                match serde_json::to_string(&response) {
                    Ok(json) => yield Ok(Event::default().data(json)),
                    Err(_) => yield Err(axum::Error::new("JSON serialization failed")),
                }

                // 模拟流式生成的词与词之间的间隔
                tokio::time::sleep(Duration::from_millis(50)).await;

                if is_final {
                    break;
                }
            }
        };

        Sse::new(stream_response).into_response()
    } else {
        // 非流式响应 - 直接执行推理并返回
        println!("执行同步推理: model={}", request.model);

        // 模拟推理计算时间
        tokio::time::sleep(Duration::from_millis(500)).await;

        let content = format!(
            "This is a direct synchronous response to '{}' using model {}. Inference is performed directly within the HTTP handler.",
            prompt.replace("\n", " "), request.model
        );

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
                    content,
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
        "mode": "handler_nested_processing",
        "info": "Inference is performed directly within the request handler"
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

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("启动直接处理模式的 OpenAI 兼容服务器...");

    // 在 main 中创建全局共享的 batch_prompts 和 batch_list
    let batch_size = 4;
    let mut prompts = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        prompts.push(BatchPrompt::new(50000));
    }
    let batch_prompts = Arc::new(Mutex::new(prompts));

    let batch_records = (0..batch_size)
        .map(|_| BatchRecord {
            sequence_index: 0,
            kv_index: 0,
            phase: Phase::Prefill_begin,
            sequence_length: 0,
            notify: Arc::new(tokio::sync::Notify::new()),
        })
        .collect::<Vec<_>>();

    let batch_list = Arc::new(Mutex::new(BatchList {
        records: batch_records.into_boxed_slice(),
        current_size: batch_size,
    }));

    let state = AppState {
        batch_prompts,
        batch_list,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/status", axum::routing::get(status))
        .with_state(state);

    let listener = TcpListener::bind("0.0.0.0:8000").await?;

    println!("服务器运行在 http://0.0.0.0:8000");
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成 (直接推理模式)");
    println!("  GET  /status - 服务器状态");

    axum::serve(listener, app).await?;
    Ok(())
}
