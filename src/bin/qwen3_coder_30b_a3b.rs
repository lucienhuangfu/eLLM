#![feature(f16)]

use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::runtime::batch_sequence::BatchSequence;
use ellm::runtime::io::load_tiktoken;
use ellm::runtime::io::ChatTemplate;
use ellm::runtime::io::SafeTensorsLoader;
use ellm::runtime::{
    BatchScheduler, Config, GenerationConfig, Phase, SequenceState, ServingRunner,
};
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

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

struct ProcessLock {
    path: PathBuf,
}

impl ProcessLock {
    fn acquire(path: impl AsRef<Path>) -> std::io::Result<Option<Self>> {
        let path = path.as_ref();
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                writeln!(file, "{}", std::process::id())?;
                Ok(Some(Self {
                    path: path.to_path_buf(),
                }))
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                let existing_pid = fs::read_to_string(path)
                    .ok()
                    .and_then(|pid| pid.trim().parse::<u32>().ok());
                if let Some(pid) = existing_pid {
                    if Path::new(&format!("/proc/{pid}")).exists() {
                        return Ok(None);
                    }
                }
                let _ = fs::remove_file(path);
                Self::acquire(path)
            }
            Err(err) => Err(err),
        }
    }
}

impl Drop for ProcessLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn main() {
    let _process_lock =
        match ProcessLock::acquire("/tmp/ellm_qwen3_coder_30b_a3b.lock").unwrap() {
            Some(lock) => lock,
            None => {
                eprintln!("qwen3_coder_30b_a3b is already running; refusing duplicate launch");
                return;
            }
        };

    let batch_size = parse_env_usize("ELLM_BATCH", 3);
    let max_output_tokens: usize = parse_env_usize("ELLM_MAX_OUTPUT_TOKENS", 128);
    let model_dir = "models/Qwen3-Coder-30B-A3B-Instruct";
    let program_start = Instant::now();

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

    let default_prompts = [
        "Write a Rust function that implements a thread-safe LRU cache.",
        "Explain how to implement a zero-copy parser in Rust using slices and references.",
        "Write a Python async function that fetches data from multiple APIs concurrently with rate limiting.",
    ];
    let env_prompt = env::var("ELLM_PROMPT").ok();
    let mut prompts = Vec::with_capacity(batch_size);
    for slot in 0..batch_size {
        if let Some(prompt) = env_prompt.as_deref() {
            prompts.push(prompt.to_string());
        } else {
            prompts.push(default_prompts[slot % default_prompts.len()].to_string());
        }
    }

    let mut all_input_lens = Vec::new();
    for prompt in &prompts {
        let rendered = chat_template
            .apply_chat_template(&[("user", prompt.as_str())], true)
            .unwrap();
        let ids = tokenizer.encode_with_special_tokens(&rendered);
        all_input_lens.push(ids.len());
    }

    let total_input: usize = all_input_lens.iter().sum();
    let max_input: usize = all_input_lens.iter().copied().max().unwrap();
    let sequence_length = max_input + max_output_tokens;
    let chunk_size = total_input + batch_size * max_output_tokens;

    let params = SafeTensorsLoader::new(model_dir)
        .unwrap()
        .load_all_weights_f16()
        .unwrap();
    f16::init_global_strict(params);
    eprintln!(
        "load_weights: {:.3}s",
        program_start.elapsed().as_secs_f64()
    );

    let position_vec = RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
        config.rope_scaling.clone(),
    )
    .forward::<f16>();

    // Force continue to max_output_tokens — disable EOS stopping.
    let eos_ids: Vec<usize> = vec![];

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
            .write_prompts(slot, &[("user", prompt.as_str())], 1.0)
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
    let requested_thread_num = parse_env_usize(
        "ELLM_THREAD_NUM",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    );
    let thread_num = physical_core_thread_limit(requested_thread_num);
    eprintln!("threads: {thread_num}");

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
    eprintln!("build_graph: {:.3}s", program_start.elapsed().as_secs_f64());

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

    // ---- force max_output_tokens cutoff after gen ----
    let sequence_length_u = sequence_length;
    let sequences_ptr_u = sequences_ptr;
    let max_output_tokens_u = max_output_tokens;

    let start = Instant::now();
    let runner = ServingRunner::new(f16::take_operator_queue(), batch_scheduler);
    runner.start();
    let elapsed = start.elapsed();

    // Force-cut each sequence to exactly max_output_tokens generated tokens
    batch_list_ref.with(|list| {
        for slot in 0..list.len() {
            let input_len = written_lengths[slot];
            // Hard cutoff: only show the first max_output_tokens generated ids
            let cut_end = (input_len + max_output_tokens_u).min(sequence_length_u);
            let gen_len = cut_end.saturating_sub(input_len);
            let ids: Vec<u32> = (input_len..cut_end)
                .map(|i| unsafe { *sequences_ptr_u.add(slot * sequence_length_u + i) as u32 })
                .collect();
            let text: String = ids
                .iter()
                .filter_map(|&tid| tokenizer.decode(vec![tid]).ok())
                .collect();
            println!("Slot {slot}: {gen_len} tokens\n{text}\n");
        }
    });

    eprintln!("generate: {:.3}s", elapsed.as_secs_f64());
    eprintln!("total: {:.3}s", program_start.elapsed().as_secs_f64());
}
