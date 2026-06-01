pub mod config;
mod handlers;
pub mod model;
pub mod model_loader;
pub mod resources;
pub mod scheduler;

use axum::{routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Semaphore};

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch_sequence::BatchSequence;
use crate::runtime::scheduling::{Phase, SequenceState, TokenCounter};

use handlers::chat_completions;

pub use config::ServerConfig;
pub use resources::{initialize_server, ServerResources};

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    index: u32,
    delta: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Clone)]
struct AppState {
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    token_counter: Arc<TokenCounter>,
    free_slots: Arc<Mutex<VecDeque<usize>>>,
    available_slots: Arc<Semaphore>,
}

fn build_app_state(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    token_counter: Arc<TokenCounter>,
) -> AppState {
    let initial_free_slots: VecDeque<usize> = batch_states.with(|batch_states_ref| {
        batch_states_ref
            .iter()
            .enumerate()
            .filter_map(|(i, record)| (record.phase == Phase::Start).then_some(i))
            .collect()
    });
    let initial_permits = initial_free_slots.len();

    AppState {
        batch_sequences,
        batch_states,
        token_counter,
        free_slots: Arc::new(Mutex::new(initial_free_slots)),
        available_slots: Arc::new(Semaphore::new(initial_permits)),
    }
}

pub async fn run(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    token_counter: Arc<TokenCounter>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("启动事件驱动的 OpenAI 兼容服务器...");

    let token_counter_task = Arc::clone(&token_counter);
    tokio::spawn(async move {
        token_counter_task.run().await;
    });

    let state = build_app_state(batch_sequences, batch_list, token_counter);

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route(
            "/status",
            axum::routing::get(|| async {
                Json(serde_json::json!({
                    "status": "running",
                    "mode": "single_threaded_background_processing",
                    "info": "Inference and HTTP server run on a single OS thread using current_thread runtime"
                }))
            }),
        )
        .with_state(state.clone());

    let listener = TcpListener::bind("0.0.0.0:8000").await?;

    println!("服务器运行在 http://0.0.0.0:8000");
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成");
    println!("  GET  /status - 服务器状态");
    println!("推理由后台 runner 订阅调度任务执行");

    axum::serve(listener, app).await?;
    Ok(())
}
