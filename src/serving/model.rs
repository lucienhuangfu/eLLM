use std::sync::Arc;
use std::time::Duration;

use crate::config::{GenerationConfig, ResolvedConfig};
use crate::mem_mgr::allocator::AlignedBox;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::executor::ExecutorPool;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{SessionMode, SlotManager};
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::shared::SharedState;
use crate::runtime::{build_batch_sequence, build_slot_state, SlotState};
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
    pub reasoning_parser_enabled: bool,
    pub tool_call_parser_enabled: bool,
    pub api_server_count: usize,
    pub session_mode: SessionMode,
    pub slot_reuse_timeout_ms: usize,
}

impl ServingConfig {
    pub fn from_resolved_config(config: &ResolvedConfig) -> Self {
        let reasoning_parser_enabled = config
            .serve
            .as_ref()
            .map(|s| s.reasoning_parser_enabled)
            .unwrap_or(true);
        let tool_call_parser_enabled = config
            .serve
            .as_ref()
            .map(|s| s.tool_call_parser_enabled)
            .unwrap_or(true);

        Self {
            model_dir: config.model.raw_config.model.clone(),
            batch_size: config.scheduler.max_num_seqs,
            sequence_length: config.model.raw_config.max_model_len.unwrap_or(128),
            chunk_size: config.scheduler.max_num_batched_tokens,
            reasoning_parser_enabled,
            tool_call_parser_enabled,
            api_server_count: config
                .serve
                .as_ref()
                .map(|s| s.api_server_count)
                .unwrap_or(2),
            session_mode: if config.scheduler.dialogue_cache_enabled {
                SessionMode::Reusable
            } else {
                SessionMode::NonReusable
            },
            slot_reuse_timeout_ms: config
                .serve
                .as_ref()
                .map(|s| s.slot_reuse_timeout_ms)
                .unwrap_or(30000),
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
    pub api_threads: usize,
    pub blocking_threads: usize,
    pub total_threads: usize,
}

pub struct ServingResources<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    pub batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    pub batch_states: Arc<SharedMut<Vec<SlotState>>>,
    pub shared_state: Arc<SharedState>,
    pub parser_options: ParserOptions,
    pub api_threads: usize,
    pub blocking_threads: usize,
    pub _sequences_box: AlignedBox<usize>,
    pub session_mode: SessionMode,
    pub slot_reuse_timeout_ms: usize,
    pub slot_manager: Arc<SlotManager<T>>,
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

fn determine_thread_config(
    generation_config: &Option<GenerationConfig>,
    api_server_count: usize,
) -> ThreadingConfig {
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
    let api_threads = api_server_count;
    let blocking_threads = (total_threads - api_threads).max(1);

    println!(
        "Total threads: {}, blocking threads: {}, api threads: {}",
        total_threads, blocking_threads, api_threads
    );

    ThreadingConfig {
        api_threads,
        blocking_threads,
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

pub fn initialize_serving_resources(
    resolved_config: &ResolvedConfig,
) -> Result<ServingResources<f16>, Box<dyn std::error::Error>> {
    let config = ServingConfig::from_resolved_config(resolved_config);
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
    let thread_config = determine_thread_config(&generation_config, config.api_server_count);
    let parser_options = ParserOptions {
        rule: ParserRule::for_model_family(&model_config.family),
        reasoning_parser: config.reasoning_parser_enabled,
        tool_call_parser: config.tool_call_parser_enabled,
    };

    let (sequences_box, batch_sequences): (AlignedBox<usize>, Arc<SharedMut<BatchSequence<f16>>>) =
        build_batch_sequence(&model_dir, config.batch_size, config.sequence_length)?;
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_states = Arc::new(SharedMut::new(build_slot_state(config.batch_size)));
    
    // Create schedule_tx channel for triggering scheduler
    let (schedule_tx, _) = tokio::sync::broadcast::channel(16);
    
    let shared_state = Arc::new(SharedState::new(
        Arc::clone(&batch_states),
        config.batch_size,
        config.chunk_size,
        thread_config.api_threads,
        schedule_tx,
    ));

    let position_vec = RotaryEmbedding::new(
        model_config.head_dim,
        model_config.rotary_dim,
        model_config.max_position_embeddings,
        model_config.rope_theta as f32,
        model_config.rope_scaling.clone(),
    )
    .forward::<f16>();
    let mut model: Model<f16> = initialize_model(
        &model_config,
        &gen_params,
        position_vec,
        config.chunk_size,
        config.batch_size,
        config.sequence_length,
    );
    model.set_thread_num(thread_config.api_threads);

    let batch_temperature_ptr =
        batch_sequences.with_mut(|batch_sequence| batch_sequence.batch_temperature.as_mut_ptr());
    let _ = model.forward(sequences_ptr, batch_temperature_ptr);

    let slot_manager = Arc::new(SlotManager::new(
        config.batch_size,
        batch_sequences.clone(),
        config.session_mode,
        config.slot_reuse_timeout_ms as u64,
    ));

    let (broadcast_sender, broadcast_receiver) = tokio::sync::broadcast::channel(8);

    let operator_queue = f16::take_operator_queue();
    let executor_pool = ExecutorPool::new(
        operator_queue,
        Arc::clone(&shared_state),
        thread_config.api_threads,
    );
    executor_pool.start(broadcast_receiver);

    let scheduler = Arc::new(Scheduler::with_mode(
        config.sequence_length,
        config.batch_size,
        config.chunk_size,
        thread_config.api_threads,
        config.chunk_size,
        Duration::from_millis(10),
        broadcast_sender,
        Arc::clone(&batch_states),
        Arc::clone(&slot_manager),
    ));
    tokio::spawn(async move {
        scheduler.run().await;
    });

    Ok(ServingResources {
        batch_sequences,
        batch_states,
        shared_state,
        parser_options,
        api_threads: thread_config.api_threads,
        blocking_threads: thread_config.blocking_threads,
        _sequences_box: sequences_box,
        session_mode: config.session_mode,
        slot_reuse_timeout_ms: config.slot_reuse_timeout_ms,
        slot_manager,
    })
}
