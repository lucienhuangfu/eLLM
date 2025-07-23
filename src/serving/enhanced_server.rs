use async_stream::stream;
use axum::{
    extract::State, http::StatusCode, response::sse::Event, response::IntoResponse, response::Sse,
    routing::post, Json, Router,
};
use crossbeam::channel::{self, Receiver, Sender};
use futures::stream::StreamExt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

// ===== LLM 工作线程相关定义 =====

#[derive(Debug)]
enum LLMTask {
    Generate {
        id: String,
        model: String,
        prompt: String,
        stream: bool,
        response_tx: Option<Sender<LLMChunk>>,
        completion_tx: Option<oneshot::Sender<String>>,
    },
    GetStatus,
    Shutdown,
}

#[derive(Debug, Clone)]
struct LLMChunk {
    request_id: String,
    content: String,
    is_final: bool,
}

#[derive(Debug)]
enum LLMResponse {
    Status {
        active_tasks: usize,
        processed_total: u64,
    },
    ShutdownAck,
}

// LLM 工作器
struct LLMWorker {
    id: usize,
    task_rx: Receiver<LLMTask>,
    response_tx: Sender<LLMResponse>,
    processed_count: u64,
    start_time: Instant,
}

impl LLMWorker {
    fn new(id: usize, task_rx: Receiver<LLMTask>, response_tx: Sender<LLMResponse>) -> Self {
        LLMWorker {
            id,
            task_rx,
            response_tx,
            processed_count: 0,
            start_time: Instant::now(),
        }
    }

    fn run(&mut self) {
        println!("LLM Worker {} 启动", self.id);

        loop {
            match self.task_rx.recv() {
                Ok(LLMTask::Generate {
                    id,
                    model,
                    prompt,
                    stream,
                    response_tx,
                    completion_tx,
                }) => {
                    println!("Worker {} 处理请求: {}", self.id, id);

                    if stream {
                        // 流式响应
                        if let Some(tx) = response_tx {
                            self.generate_streaming(&id, &model, &prompt, tx);
                        }
                    } else {
                        // 完整响应
                        if let Some(tx) = completion_tx {
                            let result = self.generate_complete(&model, &prompt);
                            let _ = tx.send(result);
                        }
                    }

                    self.processed_count += 1;
                }
                Ok(LLMTask::GetStatus) => {
                    let status = LLMResponse::Status {
                        active_tasks: 0, // 这里可以维护活跃任务计数
                        processed_total: self.processed_count,
                    };
                    let _ = self.response_tx.send(status);
                }
                Ok(LLMTask::Shutdown) => {
                    println!("Worker {} 收到关闭信号", self.id);
                    let _ = self.response_tx.send(LLMResponse::ShutdownAck);
                    break;
                }
                Err(_) => {
                    println!("Worker {} 通道关闭", self.id);
                    break;
                }
            }
        }

        println!(
            "Worker {} 结束，处理了 {} 个任务",
            self.id, self.processed_count
        );
    }

    fn generate_streaming(
        &self,
        request_id: &str,
        model: &str,
        prompt: &str,
        tx: Sender<LLMChunk>,
    ) {
        // 模拟流式文本生成
        let response = format!("AI response to '{}' using model '{}'", prompt, model);
        let words: Vec<&str> = response.split(' ').collect();

        // 逐词发送
        for (i, word) in words.into_iter().enumerate() {
            let content = if i == 0 {
                word.to_string()
            } else {
                format!(" {}", word)
            };

            let chunk = LLMChunk {
                request_id: request_id.to_string(),
                content,
                is_final: false,
            };

            if tx.send(chunk).is_err() {
                println!("流式响应通道关闭");
                return;
            }

            // 模拟生成延迟
            thread::sleep(Duration::from_millis(50));
        }

        // 发送结束标记
        let final_chunk = LLMChunk {
            request_id: request_id.to_string(),
            content: String::new(),
            is_final: true,
        };
        let _ = tx.send(final_chunk);
    }

    fn generate_complete(&self, model: &str, prompt: &str) -> String {
        // 模拟完整文本生成
        thread::sleep(Duration::from_millis(300));
        format!(
            "Complete AI response to '{}' using model '{}'",
            prompt, model
        )
    }
}

// ===== 原始的 OpenAI API 结构 =====

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
    llm_task_tx: Sender<LLMTask>,
    llm_response_rx: Arc<tokio::sync::Mutex<Receiver<LLMResponse>>>,
}

impl AppState {
    fn new(worker_count: usize) -> Self {
        let (task_tx, task_rx) = channel::unbounded::<LLMTask>();
        let (response_tx, response_rx) = channel::unbounded::<LLMResponse>();

        // 启动工作线程
        for i in 0..worker_count {
            let task_rx = task_rx.clone();
            let response_tx = response_tx.clone();

            thread::spawn(move || {
                let mut worker = LLMWorker::new(i, task_rx, response_tx);
                worker.run();
            });
        }

        AppState {
            llm_task_tx: task_tx,
            llm_response_rx: Arc::new(tokio::sync::Mutex::new(response_rx)),
        }
    }
}

// ===== API 处理函数 =====

// 格式化消息为提示
fn format_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

// 主要的聊天完成处理函数
async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let prompt = format_prompt(&request.messages);
    let request_id = format!("chatcmpl-{}", rand::thread_rng().gen::<u64>());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let is_streaming = request.stream.unwrap_or(false);

    if is_streaming {
        // 流式响应
        match handle_streaming_request(state, request, request_id, created, prompt).await {
            Ok(response) => Ok(response),
            Err(err) => Err(err),
        }
    } else {
        // 非流式响应
        match handle_non_streaming_request(state, request, request_id, created, prompt).await {
            Ok(response) => Ok(response),
            Err(err) => Err(err),
        }
    }
}

async fn handle_streaming_request(
    state: AppState,
    request: ChatCompletionRequest,
    request_id: String,
    created: u64,
    prompt: String,
) -> Result<axum::response::Response, StatusCode> {
    // 创建流式响应通道
    let (chunk_tx, chunk_rx) = channel::unbounded::<LLMChunk>();

    // 提交任务到工作线程
    let task = LLMTask::Generate {
        id: request_id.clone(),
        model: request.model.clone(),
        prompt,
        stream: true,
        response_tx: Some(chunk_tx),
        completion_tx: None,
    };

    if state.llm_task_tx.send(task).is_err() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // 创建 SSE 流
    let request_id_clone = request_id.clone();
    let model_clone = request.model.clone();

    let sse_stream = stream! {
        loop {
            match chunk_rx.recv_timeout(Duration::from_secs(30)) {
                Ok(chunk) => {
                    if chunk.is_final {
                        // 发送结束事件
                        yield Ok::<_, std::convert::Infallible>(Event::default()
                            .data(serde_json::to_string(&StreamResponse {
                                id: request_id_clone.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_clone.clone(),
                                choices: vec![StreamChoice {
                                    index: 0,
                                    delta: ChatMessage {
                                        role: "assistant".to_string(),
                                        content: "".to_string(),
                                    },
                                    finish_reason: Some("stop".to_string()),
                                }],
                            }).unwrap()));
                        break;
                    } else {
                        // 发送内容块
                        yield Ok::<_, std::convert::Infallible>(Event::default()
                            .data(serde_json::to_string(&StreamResponse {
                                id: chunk.request_id,
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_clone.clone(),
                                choices: vec![StreamChoice {
                                    index: 0,
                                    delta: ChatMessage {
                                        role: "assistant".to_string(),
                                        content: chunk.content,
                                    },
                                    finish_reason: None,
                                }],
                            }).unwrap()));
                    }
                },
                Err(_) => {
                    // 超时或错误
                    yield Ok::<_, std::convert::Infallible>(Event::default()
                        .event("error")
                        .data("Generation timeout"));
                    break;
                }
            }
        }
    };

    Ok(Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(10))
                .text("keep-alive"),
        )
        .into_response())
}

async fn handle_non_streaming_request(
    state: AppState,
    request: ChatCompletionRequest,
    request_id: String,
    created: u64,
    prompt: String,
) -> Result<axum::response::Response, StatusCode> {
    // 创建完成通道
    let (completion_tx, completion_rx) = oneshot::channel::<String>();

    // 提交任务到工作线程
    let task = LLMTask::Generate {
        id: request_id.clone(),
        model: request.model.clone(),
        prompt,
        stream: false,
        response_tx: None,
        completion_tx: Some(completion_tx),
    };

    if state.llm_task_tx.send(task).is_err() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // 等待完成
    match tokio::time::timeout(Duration::from_secs(30), completion_rx).await {
        Ok(Ok(generated_text)) => Ok((
            StatusCode::OK,
            Json(ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".to_string(),
                created,
                model: request.model,
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: generated_text,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            }),
        )
            .into_response()),
        _ => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// 状态查询端点
async fn get_status(State(state): State<AppState>) -> impl IntoResponse {
    // 请求状态
    if state.llm_task_tx.send(LLMTask::GetStatus).is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to request status".to_string(),
        );
    }

    // 等待响应 (这里简化处理)
    (StatusCode::OK, "Status: Server is running".to_string())
}

// ===== 服务器启动 =====

pub async fn run_server() -> Result<(), Box<dyn std::error::Error>> {
    println!("启动带有后台处理的 OpenAI 兼容服务器...");

    // 创建应用状态，启动2个工作线程
    let app_state = AppState::new(2);

    // 定义 API 路由
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/status", axum::routing::get(get_status))
        .with_state(app_state);

    // 启动服务器
    let listener = TcpListener::bind("0.0.0.0:8000").await?;
    println!("服务器运行在 http://{}", listener.local_addr()?);
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成");
    println!("  GET  /status - 服务器状态");

    axum::serve(listener, app).await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_server().await
}
