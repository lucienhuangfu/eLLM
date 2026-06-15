use axum::{routing::post, Json, Router};
use std::sync::Arc;
use tokio::net::TcpListener;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduling::batch_sequence::BatchSequence;
use crate::runtime::scheduling::{Scheduler, SequenceState};

use super::api::chat_completions;
use super::parser::ParserOptions;
use super::state::build_api_state;

pub async fn run(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    scheduler: Arc<Scheduler>,
    parser_options: ParserOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("启动事件驱动的 OpenAI 兼容服务器...");

    let scheduler_task = Arc::clone(&scheduler);
    tokio::spawn(async move {
        scheduler_task.run().await;
    });

    let state = build_api_state(batch_sequences, batch_list, scheduler, parser_options);

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
