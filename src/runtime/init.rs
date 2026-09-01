use std::sync::Arc;

use super::config::{
    determine_thread_config, extract_generation_params, GenerationParameters, ThreadingConfig,
};
use super::executor::executor_pool::ExecutorPool;
use crate::config::{GenerationConfig, ResolvedConfig};
use crate::mem_mgr::allocator::AlignedBox;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::model_family::config::Config;
use crate::model_family::Qwen3_model::Model;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{build_slot_sequence, SlotSequence};
use crate::runtime::session::{SessionMode, SlotManager, SlotState};
use crate::tensor::GlobalOperatorQueue;
use crate::transformer::rope::RotaryEmbedding;

use crate::runtime::loader::SafeTensorsLoader;

pub struct RuntimeContext<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    pub slot_sequences: Arc<SharedMut<SlotSequence<T>>>,
    pub scheduler: Arc<Scheduler>,
    pub slot_manager: Arc<SlotManager<T>>,
    pub thread_config: ThreadingConfig,
    pub _sequences_box: AlignedBox<usize>,
    pub session_mode: SessionMode,
    pub slot_reuse_timeout_ms: usize,
    pub max_slot_size: usize,
}

pub fn initialize_runtime(
    resolved_config: &ResolvedConfig,
    api_server_count: usize,
    batch_size: usize,
    max_slot_size: usize,
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

    let params = SafeTensorsLoader::new(model_dir)
        .and_then(|loader| loader.load_all_weights_f16())
        .map_err(|e| format!("failed to load model parameters: {}", e))?;

    println!("Loaded {} parameter tensors", params.len());
    f16::init_global_strict(params);

    let gen_params = extract_generation_params(&model_config, &generation_config);
    let thread_config = determine_thread_config(&generation_config, api_server_count);

    let (sequences_box, slot_sequences) =
        build_slot_sequence(model_dir, max_slot_size, sequence_length)?;
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_states = Arc::new(SharedMut::new(
        (0..max_slot_size)
            .map(|_| SlotState::idle())
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

    let mut model = Model::<f16>::with_sampling(
        &model_config,
        position_vec,
        chunk_size,
        sequence_length,
        max_slot_size,
        gen_params.top_k,
        gen_params.top_k_simd,
        gen_params.top_p,
        gen_params.min_p,
        gen_params.do_sample,
        gen_params.eos_token_id_list.clone(),
    );
    model.set_thread_num(thread_config.api_threads);

    let slot_temperature_ptr =
        slot_sequences.with_mut(|slot_sequence| slot_sequence.slot_temperature.as_mut_ptr());
    let _ = model.forward(sequences_ptr, slot_temperature_ptr);

    let (reasoning_parser_enabled, tool_call_parser_enabled) =
        if let Some(serve) = &resolved_config.serve {
            (
                serve.reasoning_parser_enabled,
                serve.tool_call_parser_enabled,
            )
        } else {
            (true, true)
        };

    let slot_manager = Arc::new(SlotManager::new(
        max_slot_size,
        slot_sequences.clone(),
        batch_states,
        session_mode,
        slot_reuse_timeout_ms as u64,
        reasoning_parser_enabled,
        tool_call_parser_enabled,
    ));

    let operator_queue = f16::take_operator_queue();
    let worker_pool = ExecutorPool::new(
        operator_queue,
        Arc::clone(&scheduler),
        thread_config.api_threads,
    );
    worker_pool.start();

    Ok(RuntimeContext {
        slot_sequences,
        scheduler,
        slot_manager,
        thread_config,
        _sequences_box: sequences_box,
        session_mode,
        slot_reuse_timeout_ms,
        max_slot_size,
    })
}
