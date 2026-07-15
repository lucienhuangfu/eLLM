use std::sync::Arc;
use std::time::Duration;

use crate::config::{GenerationConfig, ResolvedConfig};
use crate::mem_mgr::allocator::AlignedBox;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::executor::ExecutorPool;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{SessionMode, SlotManager};
use crate::runtime::state::batch::{build_batch_sequence, BatchSequence};
use crate::tensor::GlobalOperatorQueue;
use crate::transformer::config::Config;
use crate::transformer::model::Model;
use crate::transformer::rope::RotaryEmbedding;

/// Sampling parameters resolved from `GenerationConfig`.
#[derive(Debug, Clone)]
pub struct GenerationParameters {
    pub top_k: usize,
    pub top_k_simd: usize,
    pub top_p: f16,
    pub min_p: f16,
    pub do_sample: bool,
    pub eos_token_id_list: Vec<usize>,
}

/// Thread-pool layout: how many threads go to API vs. blocking work.
#[derive(Debug, Clone)]
pub struct ThreadingConfig {
    pub api_threads: usize,
    pub blocking_threads: usize,
    pub total_threads: usize,
}

/// All runtime handles needed to run inference, assembled in one place.
/// The serving layer wraps this with HTTP-specific config.
pub struct RuntimeContext<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    pub batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    pub scheduler: Arc<Scheduler>,
    pub slot_manager: Arc<SlotManager<T>>,
    pub thread_config: ThreadingConfig,
    pub _sequences_box: AlignedBox<usize>,
    pub session_mode: SessionMode,
    pub slot_reuse_timeout_ms: usize,
}

/// Resolve sampling parameters from an optional `GenerationConfig` and
/// the model's own `Config`.
pub fn extract_generation_params(
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

/// Detect physical cores and split threads between API handling and
/// blocking/compute work.
pub fn determine_thread_config(
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

/// Construct a `Model<f16>` with sampling from pre-resolved parameters.
pub fn initialize_model(
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

/// Bootstrap the full inference runtime from a `ResolvedConfig`.
///
/// This loads weights, builds the operator graph, warms the model up with
/// a single forward pass, and starts the executor thread pool.  The
/// returned [`RuntimeContext`] owns all shared state handles.
pub fn initialize_runtime(
    resolved_config: &ResolvedConfig,
    api_server_count: usize,
    batch_size: usize,
    sequence_length: usize,
    chunk_size: usize,
    session_mode: SessionMode,
    slot_reuse_timeout_ms: usize,
) -> Result<RuntimeContext<f16>, Box<dyn std::error::Error>> {
    let model_dir = &resolved_config.model.raw_config.model;
    println!("Loading config from: {}", model_dir);

    let model_config = Config::load_from_file(format!("{}/config.json", model_dir))
        .map_err(|e| format!("failed to load config: {}", e))?;

    let generation_config =
        GenerationConfig::load_from_file(format!("{}/generation_config.json", model_dir)).ok();

    if let Some(gen_cfg) = &generation_config {
        println!("Loaded generation config: {:?}", gen_cfg);
    }

    let params = crate::runtime::SafeTensorsLoader::new(model_dir)
        .and_then(|loader| loader.load_all_weights_f16())
        .map_err(|e| format!("failed to load model parameters: {}", e))?;

    println!("Loaded {} parameter tensors", params.len());
    f16::init_global_strict(params);

    let gen_params = extract_generation_params(&model_config, &generation_config);
    let thread_config = determine_thread_config(&generation_config, api_server_count);

    let (sequences_box, batch_sequences) =
        build_batch_sequence(model_dir, batch_size, sequence_length)?;
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_states = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| crate::runtime::session::SlotState::new_start_state())
            .collect::<Vec<_>>(),
    ));
    let scheduler = Arc::new(Scheduler::new(
        batch_size,
        chunk_size,
        thread_config.api_threads,
        Arc::clone(&batch_states),
    ));

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
        chunk_size,
        batch_size,
        sequence_length,
    );
    model.set_thread_num(thread_config.api_threads);

    let batch_temperature_ptr =
        batch_sequences.with_mut(|batch_sequence| batch_sequence.batch_temperature.as_mut_ptr());
    let _ = model.forward(sequences_ptr, batch_temperature_ptr);

    let slot_manager = Arc::new(SlotManager::new(
        batch_size,
        batch_sequences.clone(),
        batch_states,
        session_mode,
        slot_reuse_timeout_ms as u64,
    ));

    let operator_queue = f16::take_operator_queue();
    let executor_pool = ExecutorPool::new(
        operator_queue,
        Arc::clone(&scheduler),
        thread_config.api_threads,
        chunk_size,
        Duration::from_millis(10),
    );
    executor_pool.start();

    Ok(RuntimeContext {
        batch_sequences,
        scheduler,
        slot_manager,
        thread_config,
        _sequences_box: sequences_box,
        session_mode,
        slot_reuse_timeout_ms,
    })
}
