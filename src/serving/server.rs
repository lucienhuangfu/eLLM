use axum::{routing::post, Json, Router};
use std::sync::Arc;
use tokio::net::TcpListener;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{SlotManager, SlotState};
use crate::runtime::state::batch::BatchSequence;

use super::api::chat_completions;
use super::parser::ParserOptions;
use super::state::ApiState;

pub async fn run(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_list: Arc<SharedMut<Vec<SlotState>>>,
    scheduler: Arc<Scheduler>,
    parser_options: ParserOptions,
    slot_manager: Arc<SlotManager<f16>>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("启动事件驱动的 OpenAI 兼容服务器...");

    let state = ApiState {
        batch_sequences,
        batch_states: batch_list,
        scheduler,
        parser_options,
        slot_manager,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route(
            "/status",
            axum::routing::get(|| async {
                Json(serde_json::json!({
                    "status": "running",
                    "mode": "inlined_scheduler",
                    "info": "Scheduler is inlined in worker loop, executed by leader thread"
                }))
            }),
        )
        .with_state(state.clone());

    let listener = TcpListener::bind("0.0.0.0:8000").await?;

    println!("服务器运行在 http://0.0.0.0:8000");
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成");
    println!("  GET  /status - 服务器状态");
    println!("调度由 leader worker 线程内联执行");

    axum::serve(listener, app).await?;
    Ok(())
}
