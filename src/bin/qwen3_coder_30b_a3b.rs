#![feature(f16)]

use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::runtime::batch_sequence::BatchSequence;
use ellm::runtime::io::ChatTemplate;
use ellm::runtime::io::SafeTensorsLoader;
use ellm::runtime::io::load_tiktoken;
use ellm::runtime::{
    BatchScheduler, Config, GenerationConfig, Phase, ScheduleTask, SequenceState, ServingRunner,
};
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;
use std::collections::HashSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn parse_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn log_timing(label: &str, start: Instant) {
    eprintln!(
        "{label}: {:.3}s ts_ms={}",
        start.elapsed().as_secs_f64(),
        unix_timestamp_ms()
    );
}

fn physical_core_thread_limit(requested_thread_num: usize) -> usize {
    let physical_core_count = physical_core_ids().len();

    if physical_core_count == 0 {
        requested_thread_num.max(1)
    } else {
        requested_thread_num.min(physical_core_count).max(1)
    }
}

fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn physical_core_ids() -> Vec<core_affinity::CoreId> {
    let all_core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if all_core_ids.is_empty() {
        return Vec::new();
    }

    let mut seen = HashSet::new();
    let mut physical = Vec::new();
    for core_id in &all_core_ids {
        let topology_dir = format!("/sys/devices/system/cpu/cpu{}/topology", core_id.id);
        let package_id = read_trimmed(format!("{topology_dir}/physical_package_id"));
        let core_index = read_trimmed(format!("{topology_dir}/core_id"));
        let Some(package_id) = package_id else {
            continue;
        };
        let Some(core_index) = core_index else {
            continue;
        };
        if seen.insert((package_id, core_index)) {
            physical.push(*core_id);
        }
    }

    if physical.is_empty() {
        all_core_ids
            .into_iter()
            .enumerate()
            .filter_map(|(index, core_id)| (index % 2 == 0).then_some(core_id))
            .collect()
    } else {
        physical
    }
}

fn runner_core_ids(thread_num: usize, allow_logical_threads: bool) -> Vec<core_affinity::CoreId> {
    let all_core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if allow_logical_threads {
        // Stack physical cores first, then HT siblings, so that
        // threads 0..N_phys get dedicated physical cores.
        let physical = physical_core_ids();
        let logical: Vec<_> = all_core_ids
            .into_iter()
            .filter(|c| !physical.contains(c))
            .collect();
        physical
            .into_iter()
            .chain(logical)
            .take(thread_num)
            .collect()
    } else {
        physical_core_ids().into_iter().take(thread_num).collect()
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
    let _process_lock = match ProcessLock::acquire("/tmp/ellm_qwen3_coder_30b_a3b.lock").unwrap() {
        Some(lock) => lock,
        None => {
            eprintln!("qwen3_coder_30b_a3b is already running; refusing duplicate launch");
            return;
        }
    };

    let batch_size = parse_env_usize("ELLM_BATCH", 1);
    let max_output_tokens: usize = parse_env_usize("ELLM_MAX_OUTPUT_TOKENS", 512);
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
    let env_prompt = if let Ok(prompt_file) = env::var("ELLM_PROMPT_FILE") {
        Some(std::fs::read_to_string(prompt_file).expect("failed to read ELLM_PROMPT_FILE"))
    } else if let Some(repeat) = env::var("ELLM_PROMPT_REPEAT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        Some("hello ".repeat(repeat))
    } else {
        env::var("ELLM_PROMPT").ok()
    };
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
        .load_all_weights_f16_parallel()
        .unwrap();
    f16::init_global_strict(params);
    log_timing("load_weights", program_start);

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
    let requested_thread_num = parse_env_usize("ELLM_THREAD_NUM", 48);
    let allow_logical_threads = parse_env_bool("ELLM_ALLOW_LOGICAL_THREADS", true);
    let thread_num = if allow_logical_threads {
        requested_thread_num.max(1)
    } else {
        physical_core_thread_limit(requested_thread_num)
    };
    eprintln!("threads: {thread_num}");
    let pinned_core_ids = runner_core_ids(thread_num, allow_logical_threads);
    if pinned_core_ids.len() == thread_num {
        let cpu_ids = pinned_core_ids
            .iter()
            .map(|core_id| core_id.id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        eprintln!("runner_affinity: {cpu_ids}");
    } else {
        eprintln!("runner_affinity: disabled");
    }

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
    log_timing("build_graph", program_start);

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

    let mut batch_scheduler = BatchScheduler::new(sequence_length, batch_size, thread_num);
    batch_scheduler
        .batch_list
        .with_mut(|list| *list = batch_list);
    let batch_list_ref = Arc::clone(&batch_scheduler.batch_list);
    let (task_sender, _) = tokio::sync::broadcast::channel(8);
    let sizes = batch_scheduler.schedule_batch();
    let mut task = ScheduleTask::new(
        sizes.0,
        sizes.1,
        batch_scheduler.prefill_list.clone(),
        batch_scheduler.decode_list.clone(),
        1,
    )
    .with_thread_count(thread_num);

    // ---- force max_output_tokens cutoff after gen ----
    let sequence_length_u = sequence_length;
    let sequences_ptr_u = sequences_ptr;
    let max_output_tokens_u = max_output_tokens;

    let start = Instant::now();
    let task_in_flight = Arc::new(AtomicBool::new(false));
    let runner = ServingRunner::new(
        f16::take_operator_queue(),
        Arc::clone(&batch_list_ref),
        task_sender.clone(),
    )
    .with_runner_count(thread_num)
    .with_task_in_flight(Arc::clone(&task_in_flight));
    let pinned_core_ids = Arc::new(pinned_core_ids);
    let runner_handle = std::thread::spawn(move || {
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        builder.worker_threads(thread_num).enable_all();
        if pinned_core_ids.len() == thread_num {
            let next_core = Arc::new(AtomicUsize::new(0));
            let pin_core_ids = Arc::clone(&pinned_core_ids);
            builder.on_thread_start(move || {
                let index = next_core.fetch_add(1, Ordering::Relaxed);
                if let Some(core_id) = pin_core_ids.get(index % pin_core_ids.len()) {
                    core_affinity::set_for_current(*core_id);
                }
            });
        }
        let rt = builder.build().unwrap();
        rt.block_on(runner.start());
    });

    // Send prefill task — all 48 threads pick it up (thread_count=48).
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
    ellm::kernel::x86_64::f16_512::flash_attention::reset_attention_kernel_profile();
    task_in_flight.store(true, Ordering::Release);
    loop {
        match task_sender.send(task.clone()) {
            Ok(_) => break,
            Err(err) => {
                task = err.0;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }

    // Decode loop: keep scheduling until all sequences are done
    let mut generated_count = 0usize;
    loop {
        // Wait for current task to complete
        while task_in_flight.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }

        generated_count += 1;
        if generated_count == 1 {
            log_timing("first_token", start);
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512fp16"))]
            ellm::kernel::x86_64::f16_512::flash_attention::print_attention_kernel_profile(
                "first_token_prefill",
            );
        }

        // Check if all sequences are finished
        let all_done = batch_scheduler.batch_list.with(|list| {
            list.iter().all(|s| matches!(s.phase, Phase::Eos))
                || generated_count > max_output_tokens_u
        });
        if all_done {
            break;
        }

        // Schedule next batch (decode step)
        let sizes = batch_scheduler.schedule_batch();
        if sizes.1 == 0 {
            break;
        }
        let decode_task = ScheduleTask::new(
            sizes.0,
            sizes.1,
            batch_scheduler.prefill_list.clone(),
            batch_scheduler.decode_list.clone(),
            1,
        )
        .with_thread_count(thread_num);

        task_in_flight.store(true, Ordering::Release);
        loop {
            match task_sender.send(decode_task.clone()) {
                Ok(_) => break,
                Err(err) => {
                    let _ = err.0;
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        }
    }

    // Signal runner to stop
    drop(task_sender);
    // Wait for final task to complete
    while task_in_flight.load(Ordering::Acquire) {
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
    let _ = runner_handle.join();
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

    eprintln!(
        "generate: {:.3}s ts_ms={}",
        elapsed.as_secs_f64(),
        unix_timestamp_ms()
    );
    log_timing("total", program_start);
}
