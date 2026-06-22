#![feature(f16)]

use ellm::mem_mgr::allocator::AlignedBox;
use ellm::operators::operator::Operator;
use ellm::operators::send_sync_ptr::SharedMut;
use ellm::operators::testing::FakeEcho;
use ellm::runtime::{
    BatchSequence, ExecutorPool, SequenceState, SessionMode, SharedState, SlotManager,
};
use ellm::serving;
use ellm::serving::parser::{ParserOptions, ParserRule};
use ellm::transformer::config::ModelFamily;
use std::sync::Arc;
use std::time::Duration;

fn build_sequence_state(batch_size: usize) -> Vec<SequenceState> {
    (0..batch_size)
        .map(|_| SequenceState::new_start_state())
        .collect()
}

fn build_fake_runner(batch_states: Arc<SharedMut<Vec<SequenceState>>>) -> ExecutorPool<f16> {
    let operator_queue = vec![Operator::FakeEcho(FakeEcho)];
    let shared_state = Arc::new(SharedState::new(Arc::clone(&batch_states), 4, 64, 1));
    ExecutorPool::new(operator_queue, shared_state).with_thread_count(1)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting fake server for runtime + serving integration test...");

    let model_dir = "models/Qwen3-Coder-30B-A3B-Instruct";
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
    let slot_manager = Arc::new(SlotManager::new(
        batch_size,
        batch_sequences.clone(),
        SessionMode::NonReusable,
    ));
    let shared_state = Arc::new(SharedState::new(
        Arc::clone(&batch_states),
        batch_size,
        64,
        1,
    ));

    let _executor_pool = build_fake_runner(batch_states.clone());

    let parser_options = ParserOptions::new(ParserRule::for_model_family(&ModelFamily::Qwen));

    serving::run(
        batch_sequences,
        batch_states,
        shared_state,
        parser_options,
        SessionMode::NonReusable,
    )
    .await?;
    Ok(())
}
