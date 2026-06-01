mod bootstrap;
mod handlers;
mod types;

use axum::{routing::post, Json, Router};
use std::f16;
use std::sync::Arc;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch_sequence::BatchSequence;
use crate::runtime::{SequenceState, TokenCounter};

use handlers::chat_completions;

pub async fn run(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    token_counter: Arc<TokenCounter>,
) -> Result<(), Box<dyn std::error::Error>> {
    bootstrap::print_booting_message();

    let token_counter_task = Arc::clone(&token_counter);
    tokio::spawn(async move {
        token_counter_task.run().await;
    });

    let state = bootstrap::build_app_state(batch_sequences, batch_list, token_counter);

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

    let listener = bootstrap::bind_listener().await?;

    bootstrap::print_startup_info();

    axum::serve(listener, app).await?;
    Ok(())
}
