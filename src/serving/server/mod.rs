use axum::{routing::post, Json, Router};
use std::sync::Arc;

use crate::common::send_sync_ptr::SharedMut;
use crate::runtime::inference::state::SequenceState;
use crate::serving::batch_sequence::BatchSequence;

mod bootstrap;

pub mod handlers;
pub mod types;

use bootstrap::AppState;

use handlers::chat_completions;

pub async fn run(
    batch_sequences: Arc<SharedMut<BatchSequence>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    bootstrap::print_booting_message();

    let state = bootstrap::build_app_state(batch_sequences, batch_list);

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
