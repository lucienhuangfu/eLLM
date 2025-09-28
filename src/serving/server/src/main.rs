use async_stream::stream;
use axum::{
    extract::State, http::StatusCode, response::sse::Event, response::IntoResponse, response::Sse,
    routing::post, Json, Router,
};
use crossbeam::channel::{self, Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
// ------------- 新增 import -------------
use tokenizers::Tokenizer;
use minijinja::{Environment, context};

// ===== 批处理请求定义 =====

#[derive(Debug)]
struct BatchRequest {
    id: String,
    model: String,
    prompt: String,
    input_ids: Vec<u32>, // 新增：分词后的 token ids
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

// ===== 批处理收集器 =====

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
        println!(
            "批处理调度器启动，批大小: {}, 超时: {:?}",
            batch_size, batch_timeout
        );

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
                if now.duration_since(front.timestamp) >= batch_timeout || batch.len() >= batch_size
                {
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
}

// ===== LLM 工作器 =====

struct LLMWorker {
    id: usize,
    task_rx: Receiver<LLMTask>,
    processed_batches: u64,
    processed_requests: u64,
    start_time: Instant,
}

impl LLMWorker {
    fn new(id: usize, task_rx: Receiver<LLMTask>) -> Self {
        LLMWorker {
            id,
            task_rx,
            processed_batches: 0,
            processed_requests: 0,
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
                            self.processed_batches += 1;
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
            "Worker {} 结束，处理了 {} 个批次，总共 {} 个请求",
            self.id, self.processed_batches, self.processed_requests
        );
    }

    fn process_batch(&mut self, batch: Vec<BatchRequest>) {
        let batch_size = batch.len();
        println!(
            "Worker {} 开始处理批次，包含 {} 个请求",
            self.id, batch_size
        );

        // 模拟批处理的预处理阶段
        let batch_start = Instant::now();

        // 在实际应用中，这里可以进行批量优化：
        // 1. 批量分词和编码
        // 2. 批量推理计算
        // 3. 并行解码
        // 4. 内存复用和优化

        // 批量处理所有请求
        let batch_results = self.batch_inference(&batch);

        // 分发结果给各个请求
        for (request, result) in batch.into_iter().zip(batch_results.iter()) {
            self.send_result(request, result.clone());
        }

        self.processed_requests += batch_size as u64;

        let batch_duration = batch_start.elapsed();
        println!(
            "Worker {} 完成批次处理，耗时: {:?}，平均每请求: {:?}",
            self.id,
            batch_duration,
            batch_duration / batch_size as u32
        );
    }

    fn batch_inference(&self, batch: &[BatchRequest]) -> Vec<String> {
        // 模拟批处理推理
        // 在实际实现中，这里会进行真正的批量LLM推理

        let batch_size = batch.len();
        println!("Worker {} 执行批量推理，批大小: {}", self.id, batch_size);

        // 批处理通常有更好的并行效率
        // 基础时间 + 每个请求的边际时间
        let base_time = Duration::from_millis(300);
        let per_request_time = Duration::from_millis(30);
        let total_time = base_time + per_request_time * batch_size as u32;

        thread::sleep(total_time);

        // 为每个请求生成结果
        batch.iter().map(|request| {
            format!(
                "Worker {} batch response to '{}' using model {}: This is a batch-processed response (batch size: {}). Batch processing enables better resource utilization and higher throughput.",
                self.id, request.prompt, request.model, batch_size
            )
        }).collect()
    }

    fn send_result(&self, request: BatchRequest, result: String) {
        if request.stream {
            if let Some(tx) = request.response_tx {
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
                    thread::sleep(Duration::from_millis(20)); // 批处理模式下更快的流式输出
                }
            }
        } else {
            if let Some(tx) = request.completion_tx {
                let _ = tx.send(result);
            }
        }
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
    tokenizer: Arc<Tokenizer>, // 新增：全局共享 tokenizer
    chat_template: Arc<String>,   // 新增
}

impl AppState {
    fn new(worker_count: usize, batch_size: usize, batch_timeout_ms: u64, tokenizer: Arc<Tokenizer>, chat_template: Arc<String>,) -> Self {
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

        AppState { batch_collector , tokenizer,chat_template,}
    }
}

// ===== HTTP 处理器 =====

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let request_id = format!("chatcmpl-{}", generate_id());
    let is_stream = request.stream.unwrap_or(false);

    // 新写法：使用 chat_template 渲染
    let prompt = match render_chat_template(&request.messages, &state.chat_template) {
        Ok(p) => p,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    };
     // 分词（放在 spawn_blocking 避免阻塞 tokio 线程）
    let tokenizer_arc = state.tokenizer.clone();
    let prompt_clone = prompt.clone();
    let encode_handle = tokio::task::spawn_blocking(move || encode_prompt(&*tokenizer_arc, &prompt_clone));

    let input_ids = match encode_handle.await {
        Ok(Ok(ids)) => ids,
        Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("编码任务失败: {}", e)).into_response(),
    };
    
    if is_stream {
        // 流式响应
        let (chunk_tx, chunk_rx) = channel::unbounded::<LLMChunk>();

        let batch_request = BatchRequest {
            id: request_id.clone(),
            model: request.model.clone(),
            prompt,
            input_ids,
            stream: true,
            response_tx: Some(chunk_tx),
            completion_tx: None,
            timestamp: Instant::now(),
        };

        // 添加到批处理收集器
        state.batch_collector.add_request(batch_request);

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

        let batch_request = BatchRequest {
            id: request_id.clone(),
            model: request.model.clone(),
            prompt,
            input_ids,
            stream: false,
            response_tx: None,
            completion_tx: Some(completion_tx),
            timestamp: Instant::now(),
        };

        // 添加到批处理收集器
        state.batch_collector.add_request(batch_request);

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
        "mode": "batch_processing",
        "info": "Requests are collected and processed in batches for better efficiency"
    }))
}

fn generate_id() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string()
}
// ===== 辅助函数 =====

fn encode_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>, String> {
    tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("tokenizer encode failed: {}", e))
        .map(|enc| enc.get_ids().to_vec())
}
fn render_chat_template(messages: &[ChatMessage], template_str: &str) -> Result<String, String> {
    let mut env = Environment::new();
    env.add_template("chat", template_str).map_err(|e| e.to_string())?;
    let tmpl = env.get_template("chat").map_err(|e| e.to_string())?;

    let ctx = context! {
        messages => messages,
    };

    tmpl.render(ctx).map_err(|e| e.to_string())
}



// ===== 主函数 =====

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("启动批处理模式的 OpenAI 兼容服务器...");

    let worker_count = 2; // 2个worker
    let batch_size = 4; // 每批最多4个请求
    let batch_timeout_ms = 100; // 100ms超时

    println!("批处理配置:");
    println!("  - Worker数量: {}", worker_count);
    println!("  - 批大小: {}", batch_size);
    println!("  - 批超时: {}ms", batch_timeout_ms);

    // 指定 tokenizer.json 路径（把 Qwen3 的 tokenizer.json 放在项目根）
    let tokenizer_path = "./tokenizer.json";

    // 显式错误映射
    let tokenizer = Arc::new(
        Tokenizer::from_file(tokenizer_path).map_err(|e| {
            Box::<dyn std::error::Error>::from(format!("加载 tokenizer 失败: {}", e))
        })?,
    );
    // 加载 chat_template.jinja
    let chat_template = Arc::new(
        std::fs::read_to_string("./chat_template.jinja")
            .map_err(|e| format!("加载 chat_template.jinja 失败: {}", e))?,
    );
    println!("已加载 tokenizer: {}", tokenizer_path);
    println!("已加载 chat_template: ./chat_template.jinja");

    let app_state = AppState::new
    (worker_count, batch_size, batch_timeout_ms, tokenizer.clone(),chat_template.clone(),);


    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/status", axum::routing::get(status))
        .with_state(app_state);

    let listener = TcpListener::bind("0.0.0.0:8000").await?;

    println!("服务器运行在 http://0.0.0.0:8000");
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成 (批处理模式)");
    println!("  GET  /status - 服务器状态");

    axum::serve(listener, app).await?;
    Ok(())
}