use std::sync::Arc;

use crate::mem_mgr::allocator::AlignedBox;
use crate::operators::send_sync_ptr::SharedMut;
use crate::tensor::GlobalOperatorQueue;

use crate::runtime::scheduling::{
    build_batch_sequence, build_sequence_state, SequenceState, TokenCounter,
};
use crate::runtime::Runner;

use super::config::ServerConfig;
use super::model;
use super::model_loader;
use super::scheduler;

pub struct ServerResources {
    pub batch_sequences: Arc<SharedMut<crate::runtime::batch_sequence::BatchSequence<f16>>>,
    pub batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    pub token_counter: Arc<TokenCounter>,
    pub runner: Runner<f16>,
    pub worker_threads: usize,
    pub async_threads: usize,
    pub _sequences_box: AlignedBox<usize>,
}

pub fn initialize_server(
    config: &ServerConfig,
) -> Result<ServerResources, Box<dyn std::error::Error>> {
    println!("Loading config from: {}", config.model_dir);

    let (model_config, generation_config, model_dir) =
        model_loader::load_model_config(&config.model_dir)?;
    model_loader::load_model_parameters(&model_dir)?;

    let gen_params = model_loader::extract_generation_params(&model_config, &generation_config);
    let thread_config = model_loader::determine_thread_config(&generation_config);

    let (sequences_box, batch_sequences) =
        build_batch_sequence(&model_dir, config.batch_size, config.sequence_length)?;
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_states = Arc::new(SharedMut::new(build_sequence_state(config.batch_size)));
    let (token_counter, task_sender) =
        scheduler::create_scheduling_components(config, &thread_config, Arc::clone(&batch_states));

    let position_vec = model::create_position_embeddings(&model_config);
    let mut model = model::initialize_model(
        &model_config,
        &gen_params,
        position_vec,
        config.chunk_size,
        config.batch_size,
        config.sequence_length,
    );

    let batch_temperature_ptr =
        batch_sequences.with_mut(|batch_sequence| batch_sequence.batch_temperature.as_mut_ptr());
    model::run_model_forward(&mut model, sequences_ptr, batch_temperature_ptr);

    let runner = Runner::new(
        f16::take_operator_queue(),
        Arc::clone(&batch_states),
        task_sender,
    );

    Ok(ServerResources {
        batch_sequences,
        batch_states,
        token_counter,
        runner,
        worker_threads: thread_config.worker_threads,
        async_threads: thread_config.async_threads,
        _sequences_box: sequences_box,
    })
}
