#![feature(f16)]

use ellm::common::send_sync_ptr::SharedMut;
use ellm::mem_mgr::allocator::allocate_init;
use ellm::runtime::inference::{BatchScheduler, Phase, SequenceState, ServingRunner};
use ellm::serving::tokenizer_loader::load_tiktoken;
use ellm::transformer::config::Config;
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing...");

    let sequence_length = 128;
    let batch_size = 3;
    let topk_size = 8;
    let sequence_capacity = sequence_length;

    let config =
        Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

    let tokenizer_path = "models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";
    let tokenizer_config_path = "models/Qwen3-Coder-30B-A3B-Instruct/tokenizer_config.json";
    let tokenizer = load_tiktoken(tokenizer_path, tokenizer_config_path)
        .map_err(|e| format!("failed to load tokenizer: {}", e))?;

    let fixed_prompts = [
        "Hello from a fixed runner.",
        "This path does not use server input.",
        "The sequence data is hardcoded.",
    ];

    let prompt_tokens: Vec<Vec<usize>> = fixed_prompts
        .iter()
        .map(|prompt| {
            tokenizer
                .encode_with_special_tokens(prompt)
                .into_iter()
                .map(|id| id as usize)
                .collect::<Vec<_>>()
        })
        .collect();

    let sequences = allocate_init::<usize>(sequence_capacity * batch_size, 0);
    for (batch_index, tokens) in prompt_tokens.iter().enumerate() {
        let row_start = batch_index * sequence_capacity;
        for (offset, token_id) in tokens.iter().copied().enumerate() {
            if offset >= sequence_length {
                break;
            }

            unsafe {
                sequences.add(row_start + offset).write(token_id);
            }
        }
    }

    let position_vec = RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
    )
    .forward::<f16>();
    let mut model = Model::<f16>::new(
        &config,
        position_vec,
        sequence_length,
        batch_size,
        topk_size,
    );

    let eos_id = 151643;
    let (_output_indices, _output_tensor) = model.forward(sequences, eos_id);

    let thread_num = core_affinity::get_core_ids().unwrap().len();
    let mut temperature_list = Vec::with_capacity(batch_size);
    temperature_list.resize(batch_size, 1.0f16);
    for (temperature, value) in temperature_list.iter_mut().zip([0.7f16, 0.9f16, 1.0f16]) {
        *temperature = value;
    }

    let mut batch_scheduler: BatchScheduler =
        BatchScheduler::new(sequence_length, batch_size, thread_num);
    let mut batch_list = Vec::with_capacity(batch_size);
    batch_list.extend(prompt_tokens.iter().map(|tokens| SequenceState {
        filling_length: tokens.len().min(sequence_length),
        sequence_index: 0,
        kv_index: 0,
        phase: Phase::Prefill,
        notify: Arc::new(tokio::sync::Notify::new()),
    }));
    batch_scheduler.batch_list = Arc::new(SharedMut::new(batch_list));

    let runner = ServingRunner::with_temperature_list(
        model.ctx.take_operator_queue(),
        batch_scheduler,
        temperature_list,
    );
    runner.start();
    Ok(())
}
