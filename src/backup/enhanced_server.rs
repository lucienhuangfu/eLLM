use async_stream::stream;
use axum::{
    extract::State, http::StatusCode, response::sse::Event, response::IntoResponse, response::Sse,
    routing::post, Json, Router,
};
use crossbeam::channel::{self, Receiver, Sender};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

// ===== LLM 工作线程相关定义 =====

#[derive(Debug, Clone)]
struct BatchRequest {
    id: String,
    model: String,
    prompt: String,
    stream: bool,
    response_tx: Option<Sender<LLMChunk>>,
    completion_tx: Option<oneshot::Sender<String>>,
    timestamp: Instant,
}

#[derive(Debug)]
enum LLMTask {
    ProcessBatch(Vec<BatchRequest>),
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

// 批处理收集器
struct BatchCollector {
    pending_requests: Arc<Mutex<VecDeque<BatchRequest>>>,
    task_tx: Sender<LLMTask>,
    batch_size: usize,
    batch_timeout: Duration,
}

impl BatchCollector {
    fn new(task_tx: Sender<LLMTask>, batch_size: usize, batch_timeout_ms: u64) -> Self {
        let collector = BatchCollector {
            pending_requests: Arc::new(Mutex::new(VecDeque::new())),
            task_tx,
            batch_size,
            batch_timeout: Duration::from_millis(batch_timeout_ms),
        };

        // 启动批处理调度线程
        let pending_requests = collector.pending_requests.clone();
        let task_tx = collector.task_tx.clone();
        let batch_size = collector.batch_size;
        let batch_timeout = collector.batch_timeout;
        
        thread::spawn(move || {
            Self::batch_scheduler(pending_requests, task_tx, batch_size, batch_timeout);
        });

        collector
    }

    fn add_request(&self, request: BatchRequest) {
        let mut pending = self.pending_requests.lock().unwrap();
        pending.push_back(request);
        
        // 如果达到批大小，立即触发处理
        if pending.len() >= self.batch_size {
            let batch: Vec<_> = pending.drain(..self.batch_size).collect();
            drop(pending); // 释放锁
            
            if let Err(e) = self.task_tx.send(LLMTask::ProcessBatch(batch)) {
                println!("发送批处理任务失败: {:?}", e);
            }
        }
    }

    fn batch_scheduler(
        pending_requests: Arc<Mutex<VecDeque<BatchRequest>>>,
        task_tx: Sender<LLMTask>,
        batch_size: usize,
        batch_timeout: Duration,
    ) {
        println!("批处理调度器启动，批大小: {}, 超时: {:?}", batch_size, batch_timeout);
        
        loop {
            thread::sleep(batch_timeout);
            
            let mut pending = match pending_requests.lock() {
                Ok(p) => p,
                Err(_) => break,
            };
            
            if pending.is_empty() {
                continue;
            }
            
            // 检查是否有超时的请求
            let now = Instant::now();
            let mut batch = Vec::new();
            
            while let Some(front) = pending.front() {
                if now.duration_since(front.timestamp) >= batch_timeout || batch.len() >= batch_size {
                    if let Some(request) = pending.pop_front() {
                        batch.push(request);
                    }
                } else {
                    break;
                }
                
                if batch.len() >= batch_size {
                    break;
                }
            }
            
            if !batch.is_empty() {
                drop(pending); // 释放锁
                println!("调度器发送批处理任务，批大小: {}", batch.len());
                if let Err(e) = task_tx.send(LLMTask::ProcessBatch(batch)) {
                    println!("发送批处理任务失败: {:?}", e);
                    break;
                }
            }
        }
        
        println!("批处理调度器退出");
    }

// LLM 工作器 - 批处理版本
struct LLMWorker {
    id: usize,
    task_rx: Receiver<LLMTask>,
    processed_count: u64,
    start_time: Instant,
}

impl LLMWorker {
    fn new(id: usize, task_rx: Receiver<LLMTask>) -> Self {
        LLMWorker {
            id,
            task_rx,
            processed_count: 0,
            start_time: Instant::now(),
        }
    }

    fn run(&mut self) {
        println!("LLM Worker {} 启动 (批处理模式)", self.id);

        loop {
            match self.task_rx.recv() {
                Ok(task) => {
                    match task {
                        LLMTask::ProcessBatch(batch) => {
                            println!("Worker {} 处理批次，大小: {}", self.id, batch.len());
                            self.process_batch(batch);
                            self.processed_count += 1;
                        }
                        LLMTask::GetStatus => {
                            // 状态查询不需要处理，会有专门的处理逻辑
                        }
                        LLMTask::Shutdown => {
                            println!("Worker {} 收到关闭信号", self.id);
                            break;
                        }
                    }
                }
                Err(_) => {
                    println!("Worker {} 通道关闭", self.id);
                    break;
                }
            }
        }

        println!(
            "Worker {} 结束，处理了 {} 个批次",
            self.id, self.processed_count
        );
    }

    fn process_batch(&self, batch: Vec<BatchRequest>) {
        println!("Worker {} 开始处理批次，包含 {} 个请求", self.id, batch.len());
        
        // 批量处理 - 可以进行批量优化，如并行计算、内存复用等
        for (idx, request) in batch.iter().enumerate() {
            println!("Worker {} 处理批次中的请求 {}/{}: {}", 
                self.id, idx + 1, batch.len(), request.id);
            
            // 模拟批处理的LLM推理
            let result = self.process_llm_request(&request);
            
            // 发送结果
            if request.stream {
                if let Some(ref tx) = request.response_tx {
                    // 分段发送结果
                    let words: Vec<&str> = result.split_whitespace().collect();
                    for (i, word) in words.iter().enumerate() {
                        let chunk = LLMChunk {
                            request_id: request.id.clone(),
                            content: format!("{} ", word),
                            is_final: i == words.len() - 1,
                        };
                        if tx.send(chunk).is_err() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(30)); // 批处理模式下更快的流式输出
                    }
                }
            } else {
                if let Some(ref tx) = request.completion_tx {
                    let _ = tx.send(result);
                }
            }
        }
        
        println!("Worker {} 完成批次处理", self.id);
    }

    fn process_llm_request(&self, request: &BatchRequest) -> String {
        // 在批处理模式下，可以进行更多优化
        // 比如并行处理、内存复用、模型状态共享等
        
        // 模拟批处理优化后的处理时间（比单个处理更高效）
        let base_time = Duration::from_millis(200); // 批处理基础时间
        let per_request_time = Duration::from_millis(50); // 每个请求的增量时间
        let processing_time = base_time + per_request_time;
        
        thread::sleep(processing_time);

        format!(
            "Worker {} batch response to '{}' using model {}: This is a batch-processed response from the LLM. Batch processing enables better resource utilization and throughput.",
            self.id, request.prompt, request.model
        )
    }
}

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
    batch_collector: Arc<BatchCollector>,
}

impl AppState {
    fn new(worker_count: usize, batch_size: usize, batch_timeout_ms: u64) -> Self {
        let (task_tx, task_rx) = channel::unbounded::<LLMTask>();

        // 创建批处理收集器
        let batch_collector = Arc::new(BatchCollector::new(task_tx, batch_size, batch_timeout_ms));

        // 启动workers，所有worker共享同一个任务队列
        for i in 0..worker_count {
            let worker_task_rx = task_rx.clone();
            thread::spawn(move || {
                let mut worker = LLMWorker::new(i, worker_task_rx);
                worker.run();
            });
        }

        AppState {
            batch_collector,
        }
    }
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

    if is_stream {
        // 流式响应
        let (chunk_tx, chunk_rx) = channel::unbounded::<LLMChunk>();

        let task = LLMTask::Generate {
            id: request_id.clone(),
            model: request.model.clone(),
            prompt,
            stream: true,
            response_tx: Some(chunk_tx),
            completion_tx: None,
        };

        if state.llm_task_tx.send(task).is_err() {
            return (StatusCode::INTERNAL_SERVER_ERROR, "Worker unavailable").into_response();
        }

        let stream_response = stream! {
            loop {
                match chunk_rx.recv() {
                    Ok(chunk) => {
                        let response = StreamResponse {
                            id: chunk.request_id.clone(),
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
                                    content: chunk.content,
                                },
                                finish_reason: if chunk.is_final { Some("stop".to_string()) } else { None },
                            }],
                        };

                        match serde_json::to_string(&response) {
                            Ok(json) => yield Ok(Event::default().data(json)),
                            Err(_) => yield Err(axum::Error::new("JSON serialization failed")),
                        }

                        if chunk.is_final {
                            break;
                        }
                    }
                    Err(_) => {
                        yield Err(axum::Error::new("Stream closed"));
                        break;
                    }
                }
            }
        };

        Sse::new(stream_response).into_response()
    } else {
        // 非流式响应
        let (completion_tx, completion_rx) = oneshot::channel::<String>();

        let task = LLMTask::Generate {
            id: request_id.clone(),
            model: request.model.clone(),
            prompt,
            stream: false,
            response_tx: None,
            completion_tx: Some(completion_tx),
        };

        if state.llm_task_tx.send(task).is_err() {
            return (StatusCode::INTERNAL_SERVER_ERROR, "Worker unavailable").into_response();
        }

        match completion_rx.await {
            Ok(content) => {
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
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Task failed").into_response(),
        }
    }
}

async fn status() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "running",
        "workers": "shared task queue"
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
    println!("启动共享任务队列的 OpenAI 兼容服务器...");

    let worker_count = 2; // 2个worker共享任务队列
    let app_state = AppState::new(worker_count);

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/status", axum::routing::get(status))
        .with_state(app_state);

    let listener = TcpListener::bind("0.0.0.0:8000").await?;

    println!("服务器运行在 http://0.0.0.0:8000");
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成");
    println!("  GET  /status - 服务器状态");

    axum::serve(listener, app).await?;
    Ok(())
}
