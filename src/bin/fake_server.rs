#![feature(f16)]

use ellm::operators::operator::Operator;
use ellm::operators::send_sync_ptr::SharedMut;
use ellm::operators::testing::FakeEcho;
use ellm::runtime::{
    build_batch_sequence, ExecutorPool, Scheduler, SessionMode, SlotManager,
    SlotState,
};
use ellm::serving;
use ellm::serving::parser::{ParserOptions, ParserRule};
use ellm::transformer::config::ModelFamily;
use std::sync::Arc;
use std::time::Duration;

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
    parser_options: ParserOptions,
    slot_manager: Arc<SlotManager<f16>>,
    sequences_ptr: *mut usize,
    sequence_length: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = batch_states.with(|list| list.len());
    let scheduler = Arc::new(Scheduler::new(batch_size, 64, 4, Arc::clone(&batch_states)));

    let fake_echo = FakeEcho::new(sequences_ptr, sequence_length, 151643);
    let operator_queue = vec![Operator::<f16>::FakeEcho(fake_echo)];
    let executor_pool = ExecutorPool::new(
        operator_queue,
        Arc::clone(&scheduler),
        4,
        64,
        Duration::from_millis(10),
    );
    executor_pool.start();

    serving::run(scheduler, parser_options, slot_manager).await?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting fake server for runtime + serving integration test...");

    let model_dir = "models/MiniMax-M2.5";
    let sequence_length = 256usize;
    let batch_size = 4usize;

    let (sequences_box, batch_sequences) =
        build_batch_sequence(model_dir, batch_size, sequence_length)?;
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_states = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect::<Vec<_>>(),
    ));

    let slot_manager = Arc::new(SlotManager::new(
        batch_size,
        batch_sequences.clone(),
        Arc::clone(&batch_states),
        SessionMode::NonReusable,
        600000,
    ));

    let parser_options = ParserOptions::new(ParserRule::for_model_family(&ModelFamily::MiniMaxM2));

    let rt = create_runtime()?;

    rt.block_on(async move {
        run_server(
            batch_states,
            parser_options,
            slot_manager,
            sequences_ptr,
            sequence_length,
        )
        .await
    })?;

    Ok(())
}
