#![feature(f16)]

use ellm::common::send_sync_ptr::SharedMut;
use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::runtime::batch_sequence::BatchSequence;
use ellm::runtime::model_loader::SafeTensorsLoader;
use ellm::runtime::{
    BatchScheduler, Config, GenerationConfig, Phase, SequenceState, ServingRunner,
};
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing...");

    let batch_size = 3;
    let chunk_size = 64;

    let model_dir = "models/Qwen3-Coder-30B-A3B-Instruct";
    let config = Config::load_from_file(format!("{}/config.json", model_dir)).unwrap();
    let generation_config =
        GenerationConfig::load_from_file(format!("{}/generation_config.json", model_dir)).ok();

    if let Some(gen_cfg) = &generation_config {
        println!("Loaded generation config: {:?}", gen_cfg);
    }

    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tokenizer_config_path = format!("{}/tokenizer_config.json", model_dir);
    let chat_template_path = format!("{}/chat_template.jinja", model_dir);

    let sequence_length = 32;
    let top_k = 8;

    let fixed_prompts = [
        "Hello from a fixed runner.",
        "This path does not use server input.",
        "The sequence data is hardcoded.",
    ];
    let params = SafeTensorsLoader::new(model_dir)
        .and_then(|loader| loader.load_all_weights_f16())
        .map_err(|e| format!("failed to load model parameters: {}", e))?;
    println!("Loaded {} parameter tensors", params.len());
    f16::init_global_strict(params);

    // Allocate sequences buffer with sufficient size
    let sequences_capacity = config.max_position_embeddings * batch_size;
    let sequences_box = AlignedBox::allocate_init(sequences_capacity, 0);
    let sequences_ptr = sequences_box.as_mut_ptr();

    let mut batch_seq = BatchSequence::<f16>::new(
        sequences_ptr,
        batch_size,
        sequence_length,
        &tokenizer_path,
        &tokenizer_config_path,
        &chat_template_path,
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
        chunk_size,
        sequence_length,
        batch_size,
        top_k,
        eos_id,
    );

    // Run the model forward pass to populate the operator queue
    let (output_indices, output_tensor) =
        model.forward(sequences_ptr, batch_seq.batch_temperature.as_mut_ptr());

    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let thread_num = core_ids.len().max(1);
    let mut batch_scheduler: BatchScheduler =
        BatchScheduler::new(sequence_length, batch_size, thread_num);
    let mut batch_list = Vec::with_capacity(batch_size);
    batch_list.extend(
        written_lengths
            .iter()
            .enumerate()
            .map(|(_, &len)| SequenceState {
                filling_length: len.min(sequence_length),
                sequence_index: 0,
                kv_index: 0,
                phase: Phase::Prefill,
                notify: Arc::new(tokio::sync::Notify::new()),
            }),
    );
    batch_scheduler.batch_list = Arc::new(SharedMut::new(batch_list));
    let batch_list_ref = Arc::clone(&batch_scheduler.batch_list);

    println!("Starting serving runner...");
    let runner = ServingRunner::new(f16::take_operator_queue(), batch_scheduler);
    runner.start();

    println!("\n=== Generated Output ===");
    let _ = (output_indices, output_tensor);
    batch_list_ref.with(|list| {
        for (slot, record) in list.iter().enumerate() {
            let generated_begin = written_lengths[slot].min(sequence_length);
            let generated_end = record.kv_index.min(sequence_length);
            let token_ids = batch_seq.token_ids(slot, generated_begin, generated_end);
            let text = batch_seq.decode_token_span(slot, generated_begin, generated_end);
            println!("Slot {} token ids: {:?}", slot, token_ids);
            println!("Slot {} ({:?}): {}", slot, record.phase, text);
        }
    });

    Ok(())
}
