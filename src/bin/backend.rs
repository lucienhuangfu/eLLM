#![feature(f16)]

use ellm::config::GenerationConfig;
use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::operators::send_sync_ptr::SharedMut;
use ellm::runtime::{
    BatchScheduler, BatchSequence, Phase, Runner, SafeTensorsLoader, SchedulingMode, SequenceState,
};
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::config::Config;
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

    let sequence_length = 128;
    let top_k = generation_config
        .as_ref()
        .and_then(|cfg| cfg.top_k)
        .unwrap_or(8);
    let thread_num = generation_config
        .as_ref()
        .map_or_else(
            || {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            },
            |cfg| cfg.thread_num(),
        )
        .max(1);
    let top_k_simd = generation_config.as_ref().map_or_else(
        || GenerationConfig::top_k_simd_static::<f16>(top_k),
        |cfg| cfg.top_k_simd::<f16>(top_k),
    );
    let top_p = generation_config
        .as_ref()
        .and_then(|cfg| cfg.top_p)
        .unwrap_or(1.0) as f32;
    let min_p: f32 = 0.0;
    let do_sample = generation_config
        .as_ref()
        .and_then(|cfg| cfg.do_sample)
        .unwrap_or(false);

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

    // Use eos token ids from generation config when available, otherwise fall back to model config.
    let eos_token_id_list = generation_config
        .as_ref()
        .and_then(|cfg| cfg.eos_token_id_list.clone())
        .unwrap_or_else(|| vec![config.eos_token_id]);

    let mut model = Model::<f16>::new(
        &config,
        position_vec,
        chunk_size,
        sequence_length,
        batch_size,
        top_k,
        thread_num,
        top_k_simd,
        top_p as f16,
        min_p as f16,
        do_sample,
        eos_token_id_list,
    );

    // Run the model forward pass to populate the operator queue
    let (_output_indices, _output_tensor) =
        model.forward(sequences_ptr, batch_seq.batch_temperature.as_mut_ptr());

    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let thread_num = core_ids.len().max(1).min(thread_num);
    let mut batch_scheduler = BatchScheduler::with_mode(
        sequence_length,
        batch_size,
        chunk_size,
        thread_num,
        SchedulingMode::ContinuousService,
    );
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
    let runner = Runner::new(f16::take_operator_queue(), batch_scheduler);
    runner.start();

    println!("\n=== Generated Output ===");
    batch_list_ref.with(|list| {
        for (slot, record) in list.iter().enumerate() {
            let text = batch_seq.decode_generated_text(slot, record);
            if !text.is_empty() {
                println!("Slot {}: {}", slot, text);
            }
        }
    });

    Ok(())
}
