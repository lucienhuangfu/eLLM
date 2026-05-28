#![feature(f16)]

use ellm::common::send_sync_ptr::SharedMut;
use ellm::config::GenerationConfig;
use ellm::mem_mgr::allocator::AlignedBox;
use ellm::mem_mgr::mem_pool::GlobalMemPool;
use ellm::runtime::batch_sequence::BatchSequence;
use ellm::runtime::model_loader::SafeTensorsLoader;
use ellm::runtime::{BatchScheduler, SequenceState, ServingRunner};
use ellm::serving;
use ellm::tensor::GlobalOperatorQueue;
use ellm::transformer::config::Config;
use ellm::transformer::model::Model;
use ellm::transformer::rope::RotaryEmbedding;
use std::env;
use std::sync::Arc;

fn parse_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn build_sequence_state(batch_size: usize) -> Vec<SequenceState> {
    (0..batch_size)
        .map(|_| SequenceState {
            filling_length: 0,
            sequence_index: usize::MAX,
            kv_index: usize::MAX,
            phase: ellm::runtime::Phase::Start,
            notify: Arc::new(tokio::sync::Notify::new()),
        })
        .collect()
}

fn build_batch_sequence(
    model_dir: &str,
    batch_size: usize,
    sequence_length: usize,
) -> Result<
    (
        AlignedBox<usize>,
        Arc<SharedMut<BatchSequence<f16>>>,
    ),
    Box<dyn std::error::Error>,
> {
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tokenizer_config_path = format!("{}/tokenizer_config.json", model_dir);
    let chat_template_path = format!("{}/chat_template.jinja", model_dir);

    let sequences_capacity = sequence_length * batch_size;
    let sequences_box = AlignedBox::allocate_init(sequences_capacity, 0);
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_sequences = BatchSequence::<f16>::new(
        sequences_ptr,
        batch_size,
        sequence_length,
        &tokenizer_path,
        &tokenizer_config_path,
        &chat_template_path,
    )
    .map_err(|e| format!("failed to create batch sequence: {}", e))?;

    Ok((sequences_box, Arc::new(SharedMut::new(batch_sequences))))
}

fn build_batch_scheduler(
    sequence_length: usize,
    batch_size: usize,
    thread_num: usize,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
) -> BatchScheduler {
    let mut batch_scheduler = BatchScheduler::new(sequence_length, batch_size, thread_num);
    batch_scheduler.batch_list = batch_states;
    batch_scheduler
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting backend server...");

    let model_dir = env::var("ELLM_MODEL_DIR")
        .unwrap_or_else(|_| "models/Qwen3-Coder-30B-A3B-Instruct".to_string());
    let batch_size = parse_env_usize("ELLM_BATCH_SIZE", 3);
    let sequence_length = parse_env_usize("ELLM_SEQUENCE_LENGTH", 128);
    let chunk_size = parse_env_usize("ELLM_CHUNK_SIZE", 64);

    let config = Config::load_from_file(format!("{}/config.json", model_dir))
        .map_err(|e| format!("failed to load config: {}", e))?;
    let generation_config =
        GenerationConfig::load_from_file(format!("{}/generation_config.json", model_dir)).ok();

    if let Some(gen_cfg) = &generation_config {
        println!("Loaded generation config: {:?}", gen_cfg);
    }

    let top_k = generation_config
        .as_ref()
        .and_then(|cfg| cfg.top_k)
        .unwrap_or(8);
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

    let params = SafeTensorsLoader::new(&model_dir)
        .and_then(|loader| loader.load_all_weights_f16())
        .map_err(|e| format!("failed to load model parameters: {}", e))?;
    println!("Loaded {} parameter tensors", params.len());
    f16::init_global_strict(params);

    let (sequences_box, batch_sequences) =
        build_batch_sequence(&model_dir, batch_size, sequence_length)?;
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_states = Arc::new(SharedMut::new(build_sequence_state(batch_size)));
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let thread_num = core_ids.len().max(1).min(requested_thread_num);
    let batch_scheduler = build_batch_scheduler(
        sequence_length,
        batch_size,
        thread_num,
        Arc::clone(&batch_states),
    );

    let position_vec = RotaryEmbedding::new(
        config.head_dim,
        config.rotary_dim,
        config.max_position_embeddings,
        config.rope_theta as f32,
        config.rope_scaling.clone(),
    )
    .forward::<f16>();

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

    let batch_temperature_ptr = batch_sequences
        .with_mut(|batch_sequence| batch_sequence.batch_temperature.as_mut_ptr());
    let _ = model.forward(sequences_ptr, batch_temperature_ptr);

    let runner = ServingRunner::new(f16::take_operator_queue(), batch_scheduler);
    let serving_batch_states = Arc::clone(&batch_states);

    std::thread::spawn(move || {
        runner.start();
    });

    serving::run(batch_sequences, serving_batch_states).await?;
    Ok(())
}
