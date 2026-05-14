#![feature(f16)]

use ellm::common::send_sync_ptr::SharedMut;
use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::runtime::batch_sequence::BatchSequence;
use ellm::runtime::{BatchScheduler, Phase, SequenceState, ServingRunner};
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::config::Config;
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;
use std::collections::HashMap;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing...");

    let sequence_chunk_size = 128;
    let batch_size = 3;
    let topk_size = 8;

    let config =
        Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();


        
    let tokenizer_path = "models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";
    let tokenizer_config_path = "models/Qwen3-Coder-30B-A3B-Instruct/tokenizer_config.json";
    let chat_template_path = "models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";

    let fixed_prompts = [
        "Hello from a fixed runner.",
        "This path does not use server input.",
        "The sequence data is hardcoded.",
    ];

    // Initialize global memory pool
    let mut params = HashMap::new();
    params.insert("dummy.key".to_string(), vec![0.0f16; 10]);
    f16::init_global(params);

    // Allocate sequences buffer with sufficient size
    let sequences_capacity = config.max_position_embeddings * batch_size;
    let sequences_box = AlignedBox::allocate_init(sequences_capacity, 0);
    let sequences_ptr = sequences_box.as_mut_ptr();

    let mut batch_seq = BatchSequence::<f16>::new(
        sequences_ptr,
        batch_size,
        config.max_position_embeddings,
        tokenizer_path,
        tokenizer_config_path,
        chat_template_path,
    )
    .map_err(|e| format!("failed to create batch sequence: {}", e))?;

    // Write fixed prompts into each slot using BatchSequence
    let mut written_lengths = Vec::with_capacity(batch_size);
    for (slot, prompt) in fixed_prompts.iter().enumerate().take(batch_size) {
        let messages: [(&str, &str); 1] = [("user", prompt)];
        let write_len = batch_seq
            .write_prompts(slot, &messages, 1.0)
            .map_err(|e| format!("failed to write prompt: {}", e))?;
        written_lengths.push(write_len);
    }

    let position_vec = RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
        config.rope_scaling.clone(),
    )
    .forward::<f16>();

    // Use eos id from model config or default to known value
    let eos_id = config.eos_token_id;

    let mut model = Model::<f16>::new(
        &config,
        position_vec,
        sequence_chunk_size,
        batch_size,
        topk_size,
        eos_id,
    );

    // Run the model forward pass to populate the operator queue
    let (_output_indices, _output_tensor) =
        model.forward(sequences_ptr, batch_seq.batch_temperature.as_mut_ptr());

    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let thread_num = core_ids.len().max(1);
    let mut batch_scheduler: BatchScheduler =
        BatchScheduler::new(sequence_chunk_size, batch_size, thread_num);
    let mut batch_list = Vec::with_capacity(batch_size);
    batch_list.extend(
        written_lengths
            .iter()
            .enumerate()
            .map(|(i, &len)| SequenceState {
                filling_length: len.min(sequence_chunk_size),
                sequence_index: i,
                kv_index: i,
                phase: Phase::Prefill,
                notify: Arc::new(tokio::sync::Notify::new()),
            }),
    );
    batch_scheduler.batch_list = Arc::new(SharedMut::new(batch_list));

    println!("Starting serving runner...");
    let runner = ServingRunner::new(f16::take_operator_queue(), batch_scheduler);
    runner.start();
    Ok(())
}
