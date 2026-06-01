use std::collections::VecDeque;
use std::f16;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::{Mutex, Semaphore};

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch_sequence::BatchSequence;
use crate::runtime::{Phase, SequenceState, TokenCounter};

#[derive(Clone)]
pub(super) struct AppState {
    pub(super) batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    pub(super) batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    pub(super) token_counter: Arc<TokenCounter>,
    pub(super) free_slots: Arc<Mutex<VecDeque<usize>>>,
    pub(super) available_slots: Arc<Semaphore>,
}

pub(super) fn build_app_state(
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

pub(super) fn print_booting_message() {
    println!("启动事件驱动的 OpenAI 兼容服务器...");
}

pub(super) async fn bind_listener() -> Result<TcpListener, std::io::Error> {
    TcpListener::bind("0.0.0.0:8000").await
}

pub(super) fn print_startup_info() {
    println!("服务器运行在 http://0.0.0.0:8000");
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成");
    println!("  GET  /status - 服务器状态");
    println!("推理由后台 runner 订阅调度任务执行");
}
