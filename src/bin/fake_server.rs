#![feature(f16)]

use ellm::common::send_sync_ptr::SharedMut;
use ellm::mem_mgr::allocator::AlignedBox;
use ellm::operators::testing::FakeEcho;
use ellm::runtime::batch_sequence::BatchSequence;
use ellm::operators::operator::Operator;
use ellm::runtime::{BatchScheduler, Phase, SequenceState, ServingRunner};
use ellm::serving;
use std::sync::Arc;

fn build_sequence_state(batch_size: usize) -> Vec<SequenceState> {
    (0..batch_size)
        .map(|_| SequenceState {
            filling_length: 0,
            sequence_index: usize::MAX,
            kv_index: usize::MAX,
            phase: Phase::Start,
            notify: Arc::new(tokio::sync::Notify::new()),
        })
        .collect()
}

fn build_fake_runner(
    sequence_length: usize,
    batch_size: usize,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
) -> ServingRunner<f16> {
    let thread_num = core_affinity::get_core_ids()
        .map(|ids| ids.len())
        .unwrap_or(1);

    let mut batch_scheduler = BatchScheduler::new(sequence_length, batch_size, thread_num);
    batch_scheduler.batch_list = batch_states;

    let operator_queue = vec![Operator::FakeEcho(FakeEcho)];
    ServingRunner::new(operator_queue, batch_scheduler)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting fake server for runtime + serving integration test...");

    let model_dir = "models/Qwen3-Coder-30B-A3B-Instruct";
    let sequence_length = 256usize;
    let batch_size = 4usize;
    let sequences = {
        let mut boxed = AlignedBox::allocate_init(sequence_length * batch_size, 0);
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
    let runner = build_fake_runner(sequence_length, batch_size, batch_states.clone());

    std::thread::spawn(move || {
        runner.start();
    });

    serving::run(batch_sequences, batch_states).await?;
    Ok(())
}
