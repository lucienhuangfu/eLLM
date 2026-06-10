use std::env;
use std::sync::Arc;
use std::time::Duration;

use crate::config::GenerationConfig;
use crate::mem_mgr::allocator::AlignedBox;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch_sequence::BatchSequence;
use crate::runtime::scheduling::{
    build_batch_sequence, build_sequence_state, BatchScheduler, ScheduleTask, SequenceState,
    TokenCounter,
};
use crate::runtime::Runner;
use crate::tensor::GlobalOperatorQueue;
use crate::transformer::config::Config;
use crate::transformer::model::Model;
use crate::transformer::rope::RotaryEmbedding;

use super::parser::{ParserOptions, ParserRule};

#[derive(Debug, Clone)]
pub struct ServingConfig {
    pub model_dir: String,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub chunk_size: usize,
    pub schedule_timeout_ms: usize,
    pub reasoning_parser_enabled: bool,
    pub tool_call_parser_enabled: bool,
}

impl ServingConfig {
    pub fn new(model_dir: String) -> Self {
        let parse_env_usize = |name: &str, default: usize| -> usize {
            env::var(name)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(default)
        };

        Self {
            model_dir,
            batch_size: parse_env_usize("ELLM_BATCH_SIZE", 3),
            sequence_length: parse_env_usize("ELLM_SEQUENCE_LENGTH", 128),
            chunk_size: parse_env_usize("ELLM_CHUNK_SIZE", 64),
            schedule_timeout_ms: parse_env_usize("ELLM_SCHEDULE_TIMEOUT_MS", 10),
            reasoning_parser_enabled: true,
            tool_call_parser_enabled: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationParameters {
    pub top_k: usize,
    pub top_k_simd: usize,
    pub top_p: f16,
    pub min_p: f16,
    pub do_sample: bool,
    pub eos_token_id_list: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ThreadingConfig {
    pub worker_threads: usize,
    pub async_threads: usize,
    pub total_threads: usize,
}

pub struct ServingResources {
    pub batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    pub batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    pub token_counter: Arc<TokenCounter>,
    pub parser_options: ParserOptions,
    pub runner: Runner<f16>,
    pub worker_threads: usize,
    pub async_threads: usize,
    pub _sequences_box: AlignedBox<usize>,
}

fn extract_generation_params(
    config: &Config,
    generation_config: &Option<GenerationConfig>,
) -> GenerationParameters {
    let top_k = generation_config
        .as_ref()
        .and_then(|cfg| cfg.top_k)
        .unwrap_or(8);

    let top_k_simd = generation_config.as_ref().map_or_else(
        || GenerationConfig::resolved_top_k_simd_static::<f16>(top_k),
        |cfg| cfg.resolved_top_k_simd::<f16>(top_k),
    );

    let top_p = generation_config
        .as_ref()
        .and_then(|cfg| cfg.top_p)
        .unwrap_or(1.0) as f16;

    let min_p: f16 = 0.0;
    let do_sample = generation_config
        .as_ref()
        .and_then(|cfg| cfg.do_sample)
        .unwrap_or(false);

    let eos_token_id_list = generation_config
        .as_ref()
        .and_then(|cfg| cfg.eos_token_id_list.clone())
        .unwrap_or_else(|| config.eos_token_ids.clone());

    GenerationParameters {
        top_k,
        top_k_simd,
        top_p,
        min_p,
        do_sample,
        eos_token_id_list,
    }
}

fn determine_thread_config(generation_config: &Option<GenerationConfig>) -> ThreadingConfig {
    let requested_thread_num = generation_config
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

    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let physical_cores = if core_ids.is_empty() {
        requested_thread_num
    } else {
        let physical_count = core_ids
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .count();
        physical_count.max(1).min(requested_thread_num)
    };
    let total_threads = physical_cores;
    let async_threads = 2;
    let worker_threads = (total_threads - async_threads).max(1);

    println!(
        "Total threads: {}, async threads: {}, worker threads: {}",
        total_threads, async_threads, worker_threads
    );

    ThreadingConfig {
        worker_threads,
        async_threads,
        total_threads,
    }
}

fn initialize_model(
    config: &Config,
    gen_params: &GenerationParameters,
    position_vec: Vec<f16>,
    chunk_size: usize,
    batch_size: usize,
    sequence_length: usize,
) -> Model<f16> {
    Model::<f16>::with_sampling(
        config,
        position_vec,
        chunk_size,
        sequence_length,
        batch_size,
        gen_params.top_k,
        gen_params.top_k_simd,
        gen_params.top_p,
        gen_params.min_p,
        gen_params.do_sample,
        gen_params.eos_token_id_list.clone(),
    )
}

fn create_scheduling_components(
    config: &ServingConfig,
    thread_config: &ThreadingConfig,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
) -> (
    Arc<TokenCounter>,
    tokio::sync::broadcast::Sender<ScheduleTask>,
) {
    let mut batch_scheduler = BatchScheduler::with_mode(
        config.sequence_length,
        config.batch_size,
        config.chunk_size,
        thread_config.worker_threads,
    );
    batch_scheduler.batch_list = Arc::clone(&batch_states);
    let batch_scheduler = Arc::new(tokio::sync::Mutex::new(batch_scheduler));
    let broadcast_capacity = thread_config.worker_threads;
    let (task_sender, _): (tokio::sync::broadcast::Sender<ScheduleTask>, _) =
        tokio::sync::broadcast::channel(broadcast_capacity);

    let token_counter = Arc::new(TokenCounter::new(
        config.chunk_size,
        Duration::from_millis(config.schedule_timeout_ms as u64),
        Arc::clone(&batch_scheduler),
        task_sender.clone(),
    ));

    (token_counter, task_sender)
}

pub fn initialize_serving_resources(
    config: &ServingConfig,
) -> Result<ServingResources, Box<dyn std::error::Error>> {
    println!("Loading config from: {}", config.model_dir);

    let model_config = Config::load_from_file(format!("{}/config.json", config.model_dir))
        .map_err(|e| format!("failed to load config: {}", e))?;
    let generation_config =
        GenerationConfig::load_from_file(format!("{}/generation_config.json", config.model_dir))
            .ok();

    if let Some(gen_cfg) = &generation_config {
        println!("Loaded generation config: {:?}", gen_cfg);
    }

    let model_dir = config.model_dir.clone();

    let params = crate::runtime::SafeTensorsLoader::new(&model_dir)
        .and_then(|loader| loader.load_all_weights_f16())
        .map_err(|e| format!("failed to load model parameters: {}", e))?;

    println!("Loaded {} parameter tensors", params.len());
    f16::init_global_strict(params);

    let gen_params = extract_generation_params(&model_config, &generation_config);
    let thread_config = determine_thread_config(&generation_config);
    let parser_options = ParserOptions {
        rule: ParserRule::for_model_family(&model_config.family),
        reasoning_parser: config.reasoning_parser_enabled,
        tool_call_parser: config.tool_call_parser_enabled,
    };

    let (sequences_box, batch_sequences) =
        build_batch_sequence(&model_dir, config.batch_size, config.sequence_length)?;
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_states = Arc::new(SharedMut::new(build_sequence_state(config.batch_size)));
    let (token_counter, task_sender) =
        create_scheduling_components(config, &thread_config, Arc::clone(&batch_states));

    let position_vec = RotaryEmbedding::new(
        model_config.head_dim,
        model_config.rotary_dim,
        model_config.max_position_embeddings,
        model_config.rope_theta as f32,
        model_config.rope_scaling.clone(),
    )
    .forward::<f16>();
    let mut model = initialize_model(
        &model_config,
        &gen_params,
        position_vec,
        config.chunk_size,
        config.batch_size,
        config.sequence_length,
    );
    model.set_thread_num(thread_config.worker_threads);

    let batch_temperature_ptr =
        batch_sequences.with_mut(|batch_sequence| batch_sequence.batch_temperature.as_mut_ptr());
    let _ = model.forward(sequences_ptr, batch_temperature_ptr);

    let runner = Runner::new(
        f16::take_operator_queue(),
        Arc::clone(&batch_states),
        task_sender,
    )
    .with_runner_count(thread_config.worker_threads)
    .with_task_in_flight(token_counter.task_in_flight());

    Ok(ServingResources {
        batch_sequences,
        batch_states,
        token_counter,
        parser_options,
        runner,
        worker_threads: thread_config.worker_threads,
        async_threads: thread_config.async_threads,
        _sequences_box: sequences_box,
    })
}
