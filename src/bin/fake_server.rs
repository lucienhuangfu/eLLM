#![feature(f16)]

use ellm::operators::operator::Operator;
use ellm::operators::send_sync_ptr::SharedMut;
use ellm::operators::testing::FakeEcho;
use ellm::runtime::{
    build_slot_sequence, ExecutorPool, Scheduler, SessionMode, SlotManager, SlotState,
};
use ellm::serving;
use std::sync::Arc;

fn create_runtime() -> Result<tokio::runtime::Runtime, Box<dyn std::error::Error>> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(4)
        .enable_all()
        .build()
        .map_err(Into::into)
}

async fn run_server(
    batch_states: Arc<SharedMut<Vec<SlotState>>>,
    slot_manager: Arc<SlotManager<f16>>,
    sequences_ptr: *mut usize,
    sequence_length: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = batch_states.with(|list| list.len());
    let scheduler = Arc::new(Scheduler::new(batch_size, 64, 4, Arc::clone(&batch_states)));

    let digit_tokens: Vec<usize> = (15..25).collect();
    let fake_echo = FakeEcho::new(sequences_ptr, sequence_length, 151643, digit_tokens);
    let operator_queue = vec![Operator::<f16>::FakeEcho(fake_echo)];
    let worker_pool = ExecutorPool::new(operator_queue, Arc::clone(&scheduler), 4);
    worker_pool.start();

    serving::run(slot_manager, "0.0.0.0", 8000).await?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting fake server for runtime + serving integration test...");

    let model_dir = "models/MiniMax-M2.5";
    let sequence_length = 256usize;
    let batch_size = 4usize;

    let (sequences_box, slot_sequences) =
        build_slot_sequence(model_dir, batch_size, sequence_length)?;
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_states = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::idle())
            .collect::<Vec<_>>(),
    ));

    let slot_manager = Arc::new(SlotManager::new(
        batch_size,
        slot_sequences.clone(),
        Arc::clone(&batch_states),
        SessionMode::NonReusable,
        600000,
        true,
        true,
    ));

    let rt = create_runtime()?;

    rt.block_on(async move {
        run_server(batch_states, slot_manager, sequences_ptr, sequence_length).await
    })?;

    Ok(())
}
