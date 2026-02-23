use axum::{routing::post, Json, Router};
use std::collections::VecDeque;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Semaphore};

use crate::common::send_sync_ptr::SharedMut;
use crate::serving::batch_sequence::BatchSequence;
use crate::serving::record::{Phase, SequenceState};

pub mod handlers;
pub mod types;

use handlers::chat_completions;

#[derive(Clone)]
pub(super) struct AppState {
    pub(super) batch_sequences: Arc<SharedMut<BatchSequence>>,
    pub(super) batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    pub(super) tokenizer: Arc<Tokenizer>,
    pub(super) free_slots: Arc<Mutex<VecDeque<usize>>>,
    pub(super) available_slots: Arc<Semaphore>,
}

pub async fn run(
    batch_sequences: Arc<SharedMut<BatchSequence>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    tokenizer: Arc<Tokenizer>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("启动单线程架构的 OpenAI 兼容服务器...");

    let initial_free_slots: VecDeque<usize> = {
        let batch_list_ref = unsafe { &*batch_list.get() };
        batch_list_ref
            .iter()
            .enumerate()
            .filter_map(|(i, record)| (record.phase == Phase::Start).then_some(i))
            .collect()
    };
    let initial_permits = initial_free_slots.len();

    let state = AppState {
        batch_sequences,
        batch_list,
        tokenizer,
        free_slots: Arc::new(Mutex::new(initial_free_slots)),
        available_slots: Arc::new(Semaphore::new(initial_permits)),
    };

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
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成 (单线程推理模式)");
    println!("  GET  /status - 服务器状态");

    println!("推理由外部 runner 提供，HTTP 仅负责接收请求与响应");

    axum::serve(listener, app).await?;
    Ok(())
}
