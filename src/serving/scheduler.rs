use std::sync::Arc;
use std::time::Duration;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduling::ScheduleTask;
use crate::serving::config::ServingConfig;
use crate::serving::model_setup::ThreadingConfig;

use crate::runtime::scheduling::{BatchScheduler, SequenceState, TokenCounter};

pub fn build_batch_scheduler(
    sequence_length: usize,
    batch_size: usize,
    chunk_size: usize,
    thread_num: usize,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
) -> BatchScheduler {
    let mut batch_scheduler =
        BatchScheduler::with_mode(sequence_length, batch_size, chunk_size, thread_num);
    batch_scheduler.batch_list = batch_states;
    batch_scheduler
}

pub fn create_scheduling_components(
    config: &ServingConfig,
    thread_config: &ThreadingConfig,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
) -> (
    Arc<TokenCounter>,
    tokio::sync::broadcast::Sender<ScheduleTask>,
) {
    let batch_scheduler = build_batch_scheduler(
        config.sequence_length,
        config.batch_size,
        config.chunk_size,
        thread_config.total_threads,
        Arc::clone(&batch_states),
    );
    let batch_scheduler = Arc::new(tokio::sync::Mutex::new(batch_scheduler));
    let broadcast_capacity = thread_config.worker_threads;
    let (task_sender, _): (tokio::sync::broadcast::Sender<ScheduleTask>, _) =
        tokio::sync::broadcast::channel(broadcast_capacity);

    let token_counter = Arc::new(TokenCounter::new(
        config.chunk_size,
        Duration::from_millis(config.schedule_timeout_ms as u64),
        Arc::clone(&batch_scheduler),
        task_sender.clone(),
    ));

    (token_counter, task_sender)
}
