use std::collections::VecDeque;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::{Mutex, Semaphore};

use crate::common::send_sync_ptr::SharedMut;
use crate::runtime::inference::state::{Phase, SequenceState};
use crate::serving::batch_sequence::BatchSequence;

#[derive(Clone)]
pub(in crate::serving::server) struct AppState {
    pub(in crate::serving::server) batch_sequences: Arc<SharedMut<BatchSequence>>,
    pub(in crate::serving::server) batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    pub(in crate::serving::server) free_slots: Arc<Mutex<VecDeque<usize>>>,
    pub(in crate::serving::server) available_slots: Arc<Semaphore>,
}

pub(super) fn build_app_state(
    batch_sequences: Arc<SharedMut<BatchSequence>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
) -> AppState {
    let initial_free_slots: VecDeque<usize> = batch_list.with(|batch_list_ref| {
        batch_list_ref
            .iter()
            .enumerate()
            .filter_map(|(i, record)| (record.phase == Phase::Start).then_some(i))
            .collect()
    });
    let initial_permits = initial_free_slots.len();

    AppState {
        batch_sequences,
        batch_list,
        free_slots: Arc::new(Mutex::new(initial_free_slots)),
        available_slots: Arc::new(Semaphore::new(initial_permits)),
    }
}

pub(super) fn print_booting_message() {
    println!("启动单线程架构的 OpenAI 兼容服务器...");
}

pub(super) async fn bind_listener() -> Result<TcpListener, std::io::Error> {
    TcpListener::bind("0.0.0.0:8000").await
}

pub(super) fn print_startup_info() {
    println!("服务器运行在 http://0.0.0.0:8000");
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成 (单线程推理模式)");
    println!("  GET  /status - 服务器状态");
    println!("推理由外部 runner 提供，HTTP 仅负责接收请求与响应");
}
