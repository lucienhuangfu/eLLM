#![feature(f16)]

use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::operators::send_sync_ptr::SharedMut;
use ellm::runtime::io::load_tiktoken;
use ellm::runtime::io::ChatTemplate;
use ellm::runtime::io::SafeTensorsLoader;
use ellm::runtime::{
    BatchSequence, Config, ExecutorPool, GenerationConfig, Phase, ScheduleTask, Scheduler,
    SessionMode, SlotManager, SlotState,
};
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;
use std::env;
use std::sync::Arc;
use std::time::Duration;

fn parse_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn physical_core_thread_limit(requested_thread_num: usize) -> usize {
    let all_core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let physical_core_count = all_core_ids
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .count();

    if physical_core_count == 0 {
        requested_thread_num.max(1)
    } else {
        requested_thread_num.min(physical_core_count).max(1)
    }
}

fn main() {
    let batch_size = 3;
    let max_output_tokens = parse_env_usize("ELLM_MAX_OUTPUT_TOKENS", 32);
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
    let sequence_length = max_input + max_output_tokens;
    let chunk_size = total_input + batch_size * max_output_tokens;

    println!("max_input={max_input} total_input={total_input} seq_len={sequence_length} chunk={chunk_size}");

    let params = SafeTensorsLoader::new(model_dir)
        .unwrap()
        .load_all_weights_f16_parallel()
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
    let top_k_simd = gen_cfg.as_ref().map_or_else(
        || GenerationConfig::resolved_top_k_simd_static::<f16>(top_k),
        |cfg| cfg.resolved_top_k_simd::<f16>(top_k),
    );
    let top_p = gen_cfg.as_ref().and_then(|g| g.top_p).unwrap_or(1.0) as f32;
    let min_p: f32 = 0.0;
    let do_sample = gen_cfg.as_ref().and_then(|g| g.do_sample).unwrap_or(false);
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let requested_thread_num = parse_env_usize(
        "ELLM_THREAD_NUM",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    );
    let thread_num = if core_ids.is_empty() {
        requested_thread_num
    } else {
        physical_core_thread_limit(requested_thread_num)
    };
    println!("Threads: {thread_num}");

    println!("Building model graph...");
    let mut model = Model::<f16>::with_sampling(
        &config,
        position_vec,
        chunk_size,
        sequence_length,
        batch_size,
        top_k,
        top_k_simd,
        top_p as f16,
        min_p as f16,
        do_sample,
        eos_ids,
    );
    model.set_thread_num(thread_num);
    let (_indices, _values) =
        model.forward(sequences_ptr, batch_seq.batch_temperature.as_mut_ptr());

    let batch_list: Vec<SlotState> = written_lengths
        .iter()
        .map(|&len| SlotState::new_prefill_state(0, len))
        .collect();
    let batch_list_arc = Arc::new(SharedMut::new(batch_list));
    let batch_seq_arc = Arc::new(SharedMut::new(batch_seq));

    let batch_scheduler = Arc::new(Scheduler::new(
        batch_size,
        chunk_size,
        thread_num,
        Arc::clone(&batch_list_arc),
    ));

    let slot_manager = Arc::new(SlotManager::new(
        batch_size,
        Arc::clone(&batch_seq_arc),
        Arc::clone(&batch_list_arc),
        SessionMode::Reusable,
        600000, // 10 minutes
    ));

    let executor_pool = ExecutorPool::new(
        f16::take_operator_queue(),
        Arc::clone(&batch_scheduler),
        thread_num,
        chunk_size,
        Duration::from_millis(10),
    );

    if batch_scheduler.schedule_batch() {
        batch_scheduler.with_task(|task| {
            // Execute prefill task
            // executor_pool.execute_task(task);
        });
    }

    println!("Starting inference with ExecutorPool...");
    let start = std::time::Instant::now();
    let max_output_tokens_u = max_output_tokens;

    // Decode loop
    let mut generated_count = 0usize;
    loop {
        generated_count += 1;

        let all_done = batch_scheduler.batch_list().with(|list| {
            list.iter().all(|s| matches!(s.phase, Phase::Eos))
                || generated_count > max_output_tokens_u
        });
        if all_done {
            break;
        }

        if !batch_scheduler.schedule_batch() {
            break;
        }

        let decode_size = batch_scheduler.with_task(|task| task.decode_size);
        if decode_size == 0 {
            break;
        }

        batch_scheduler.with_task(|task| {
            // executor_pool.execute_task(task);
        });
    }

    let elapsed = start.elapsed();
    println!("Done in {elapsed:.2?}\n");

    batch_list_arc.with(|list| {
        batch_seq_arc.with(|batch_seq| {
            for (slot, record) in list.iter().enumerate() {
                let input_len = written_lengths[slot];
                let actual_gen_len = record.kv_index.saturating_sub(input_len);
                let gen_end = record.kv_index.min(sequence_length);
                let gen_len = gen_end.saturating_sub(input_len);
                let _text_short = batch_seq.decode_token_span(slot, input_len, gen_end);
                let _ids = batch_seq.token_ids(slot, input_len, gen_end.min(input_len + 5));
            let ids: Vec<u32> = (input_len..gen_end)
                .map(|i| unsafe { *sequences_ptr.add(slot * sequence_length + i) as u32 })
                .collect();
            // Decode all tokens individually (tiktoken batch decode can fail on special tokens)
            let full_text: String = ids
                .iter()
                .filter_map(|&tid| tokenizer.decode(vec![tid]).ok())
                .collect();
            println!(
                    "Slot {slot} [{p}]: {gen_len} displayed tokens, actual_gen_len={actual_gen_len}, phase={phase:?}",
                    p = prompts[slot],
                    phase = record.phase
                );
                println!("  {full_text:?}");
                println!();
            }
        });
    });
}
