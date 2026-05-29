#![feature(f16)]

use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::operators::send_sync_ptr::SharedMut;
use ellm::runtime::batch_sequence::BatchSequence;
use ellm::runtime::chat_template::ChatTemplate;
use ellm::runtime::model_loader::SafeTensorsLoader;
use ellm::runtime::tokenizer_loader::load_tiktoken;
use ellm::runtime::{
    BatchScheduler, Config, GenerationConfig, Phase, SequenceState, ServingRunner,
};
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;
use std::sync::Arc;

fn main() {
    let batch_size = 3;
    let max_output_tokens = 32;
    let model_dir = "models/Qwen3-0.6B";

    let config = Config::load_from_file(format!("{}/config.json", model_dir)).unwrap();
    let gen_cfg =
        GenerationConfig::load_from_file(format!("{}/generation_config.json", model_dir)).ok();

    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tokenizer_config_path = format!("{}/tokenizer_config.json", model_dir);
    let chat_template_path = format!("{}/chat_template.jinja", model_dir);

    let chat_template = ChatTemplate::from_model_files(&chat_template_path, &tokenizer_config_path)
        .ok()
        .unwrap();
    let tokenizer = load_tiktoken(&tokenizer_path, &tokenizer_config_path).unwrap();

    let prompts = [
        "请用 Rust 写一个计算斐波那契数列的函数。",
        "What is the difference between stack and heap memory?",
        "Tell me a short joke about programming.",
    ];

    // Tokenize to determine sizes
    let mut all_input_lens = Vec::new();
    for prompt in &prompts {
        let rendered = chat_template
            .apply_chat_template(&[("user", *prompt)], true)
            .unwrap();
        let ids = tokenizer.encode_with_special_tokens(&rendered);
        println!("Prompt '{prompt}': {len} tokens", len = ids.len());
        all_input_lens.push(ids.len());
    }

    let total_input: usize = all_input_lens.iter().sum();
    let max_input: usize = all_input_lens.iter().copied().max().unwrap();
    let sequence_length = max_input + max_output_tokens + 64;
    let chunk_size = total_input + batch_size * max_output_tokens + 64;

    println!("max_input={max_input} total_input={total_input} seq_len={sequence_length} chunk={chunk_size}");

    let params = SafeTensorsLoader::new(model_dir)
        .unwrap()
        .load_all_weights_f16()
        .unwrap();
    println!("Loaded {} tensors", params.len());
    f16::init_global_strict(params);

    let position_vec = RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
        config.rope_scaling.clone(),
    )
    .forward::<f16>();

    let eos_ids = gen_cfg
        .as_ref()
        .and_then(|g| g.eos_token_id_list.clone())
        .filter(|ids| !ids.is_empty())
        .unwrap_or(config.eos_token_ids.clone());

    let sequences_capacity = sequence_length * batch_size;
    let sequences_box = AlignedBox::allocate_init(sequences_capacity, 0usize);
    let sequences_ptr = sequences_box.as_mut_ptr();

    let mut batch_seq = BatchSequence::<f16>::new(
        sequences_ptr,
        batch_size,
        sequence_length,
        &tokenizer_path,
        &tokenizer_config_path,
        &chat_template_path,
    )
    .unwrap();

    let mut written_lengths = Vec::new();
    for (slot, prompt) in prompts.iter().enumerate().take(batch_size) {
        let write_len = batch_seq
            .write_prompts(slot, &[("user", prompt)], 1.0)
            .unwrap();
        written_lengths.push(write_len);
    }

    let top_k = gen_cfg.as_ref().and_then(|g| g.top_k).unwrap_or(1);
    let top_p = gen_cfg.as_ref().and_then(|g| g.top_p).unwrap_or(1.0) as f32;
    let min_p: f32 = 0.0;
    let do_sample = gen_cfg.as_ref().and_then(|g| g.do_sample).unwrap_or(false);

    println!("Building model graph...");
    let mut model = Model::<f16>::with_sampling(
        &config,
        position_vec,
        chunk_size,
        sequence_length,
        batch_size,
        top_k,
        top_p as f16,
        min_p as f16,
        do_sample,
        eos_ids,
    );

    let (_indices, _values) =
        model.forward(sequences_ptr, batch_seq.batch_temperature.as_mut_ptr());

    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let thread_num = core_ids.len().max(1);
    println!("Threads: {thread_num}");

    let batch_list: Vec<SequenceState> = written_lengths
        .iter()
        .map(|&len| SequenceState {
            filling_length: len,
            sequence_index: 0,
            kv_index: 0,
            phase: Phase::Prefill,
            notify: Arc::new(tokio::sync::Notify::new()),
        })
        .collect();

    let batch_scheduler = BatchScheduler::new(sequence_length, batch_size, thread_num);
    batch_scheduler
        .batch_list
        .with_mut(|list| *list = batch_list);
    let batch_list_ref = Arc::clone(&batch_scheduler.batch_list);

    println!("Starting inference...");
    let start = std::time::Instant::now();

    let runner = ServingRunner::new(f16::take_operator_queue(), batch_scheduler);
    runner.start();

    let elapsed = start.elapsed();
    println!("Done in {elapsed:.2?}\n");

    batch_list_ref.with(|list| {
        for (slot, record) in list.iter().enumerate() {
            let input_len = written_lengths[slot];
            let gen_end = record
                .kv_index
                .min(input_len + 32.min(sequence_length - input_len));
            let gen_len = gen_end.saturating_sub(input_len);
            let text_short = batch_seq.decode_token_span(slot, input_len, gen_end);
            let ids = batch_seq.token_ids(slot, input_len, gen_end.min(input_len + 5));
            let ids: Vec<u32> = (input_len..gen_end)
                .map(|i| unsafe { *sequences_ptr.add(slot * sequence_length + i) as u32 })
                .collect();
            // Decode all tokens individually (tiktoken batch decode can fail on special tokens)
            let full_text: String = ids
                .iter()
                .filter_map(|&tid| tokenizer.decode(vec![tid]).ok())
                .collect();
            println!("Slot {slot} [{p}]: {gen_len} tokens", p = prompts[slot]);
            println!("  {full_text:?}");
            println!();
        }
    });
}
