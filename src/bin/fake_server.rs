#![feature(f16)]

use ellm::mem_mgr::allocator::AlignedBox;
use ellm::operators::operator::Operator;
use ellm::operators::send_sync_ptr::SharedMut;
use ellm::operators::testing::FakeEcho;
use ellm::runtime::{
    BatchSequence, ExecutorPool, Scheduler, SessionMode, SharedState, SlotManager, SlotState,
};
use ellm::serving;
use ellm::serving::parser::{ParserOptions, ParserRule};
use ellm::transformer::config::ModelFamily;
use std::sync::Arc;
use std::time::Duration;

fn build_sequence_state(batch_size: usize) -> Vec<SlotState> {
    (0..batch_size)
        .map(|_| SlotState::new_start_state())
        .collect()
}

fn create_runtime(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_states: Arc<SharedMut<Vec<SlotState>>>,
    parser_options: ParserOptions,
) -> Result<tokio::runtime::Runtime, Box<dyn std::error::Error>> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(4)
        .enable_all()
        .build()
        .map_err(Into::into)
}

async fn run_server(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_states: Arc<SharedMut<Vec<SlotState>>>,
    parser_options: ParserOptions,
    slot_manager: Arc<SlotManager<f16>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create broadcast channel for scheduler -> executor communication
    let (broadcast_sender, broadcast_receiver) = tokio::sync::broadcast::channel(8);

    let shared_state = Arc::new(SharedState::new(Arc::clone(&batch_states)));

    // Build and start executor pool with FakeEcho operator
    let operator_queue = vec![Operator::<f16>::FakeEcho(FakeEcho)];
    let executor_pool = ExecutorPool::new(operator_queue, Arc::clone(&shared_state), 1, slot_manager.clone(), Duration::from_millis(10));
    executor_pool.start();

    let batch_size = batch_states.with(|list| list.len());
    let sequence_length = 256usize;

    // Create scheduler with shared_state
    let scheduler = Arc::new(Scheduler::with_shared_state(
        sequence_length,
        batch_size,
        64,
        1,
        64,
        Duration::from_millis(10),
        broadcast_sender,
        Arc::clone(&batch_states),
        Arc::clone(&slot_manager),
        Arc::clone(&shared_state),
    ));
    tokio::spawn(async move {
        scheduler.run().await;
    });

    serving::run(
        batch_sequences,
        batch_states,
        shared_state,
        parser_options,
        slot_manager,
    )
    .await?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting fake server for runtime + serving integration test...");

    let model_dir = "models/MiniMax-M2.5";
    let sequence_length = 256usize;
    let batch_size = 4usize;
    let sequences = {
        let boxed = AlignedBox::allocate_init(sequence_length * batch_size, 0);
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        ptr
    };

    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tokenizer_config_path = format!("{}/tokenizer_config.json", model_dir);
    let chat_template_path = format!("{}/chat_template.jinja", model_dir);

    let batch_sequences = Arc::new(SharedMut::new(
        BatchSequence::<f16>::new(
            sequences,
            batch_size,
            sequence_length,
            tokenizer_path.as_str(),
            tokenizer_config_path.as_str(),
            chat_template_path.as_str(),
        )
        .map_err(|e| format!("Unable to initialize BatchSequence: {}", e))?,
    ));

    let batch_states = Arc::new(SharedMut::new(build_sequence_state(batch_size)));

    // Create slot manager BEFORE entering Tokio runtime
    let slot_manager = Arc::new(SlotManager::new(
        batch_size,
        batch_sequences.clone(),
        SessionMode::NonReusable,
        600000, // 10 minutes
    ));

    let parser_options = ParserOptions::new(ParserRule::for_model_family(&ModelFamily::MiniMaxM2));

    // Create Tokio runtime explicitly
    let rt = create_runtime(
        batch_sequences.clone(),
        batch_states.clone(),
        parser_options.clone(),
    )?;

    rt.block_on(async move {
        run_server(batch_sequences, batch_states, parser_options, slot_manager).await
    })?;

    Ok(())
}
