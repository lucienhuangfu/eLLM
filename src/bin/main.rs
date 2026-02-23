#![feature(f16)]

use ellm::serving::record::{Phase, SequenceState};
use ellm::common::send_sync_ptr::SharedMut;
use ellm::mem_mgr::allocator::allocate_init;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;
use ellm::qwen3_moe::rope::precompute_freqs_cis_t;
use ellm::serving::batch_sequence::BatchSequence;
use ellm::serving::runner::ServingRunner;
use ellm::serving::schedule::BatchScheduler;
use ellm::serving::server;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing...");

    let sequence_length = 128;
    let batch_size = 3;
    let topk_size = 8;

    let config =
        Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

    let position_vec = precompute_freqs_cis_t::<f16>(
        config.head_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
    );
    let mut model = Model::<f16>::new(
        &config,
        position_vec,
        sequence_length,
        batch_size,
        topk_size,
    );

    let sequence_capacity = sequence_length + 1;
    let sequences = allocate_init::<usize>(sequence_capacity * batch_size, 0);

    let tokenizer_path = "models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("无法加载分词器 {}: {}", tokenizer_path, e))?;
    let tokenizer = Arc::new(tokenizer);

    let batch_sequences = Arc::new(SharedMut::new(BatchSequence::new(
        sequences,
        batch_size,
        sequence_capacity,
    )));

    let eos_id = 151643;

    let (output_indices, output_tensor) = unsafe { model.forward(sequences, eos_id) };

    let thread_num = core_affinity::get_core_ids().unwrap().len();
    let mut batch_scheduler = BatchScheduler::new(sequence_length, batch_size, thread_num);
    let mut batch_list = Vec::with_capacity(batch_size);
    batch_list.extend((0..batch_size).map(|_| SequenceState {
        sequence_index: 0,
        kv_index: 0,
        phase: Phase::Start,
        notify: std::sync::Arc::new(tokio::sync::Notify::new()),
    }));
    batch_scheduler.batch_list = Arc::new(SharedMut::new(batch_list));

    let batch_list = batch_scheduler.batch_list.clone();
    let runner = ServingRunner::new(model.ctx.take_operator_queue(), batch_scheduler);

    std::thread::spawn(move || {
        runner.start();
    });

    server::run(batch_sequences, batch_list, tokenizer).await?;
    Ok(())
}
