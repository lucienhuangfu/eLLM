use crate::config::GenerationConfig;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::transformer::config::Config;

pub struct GenerationParameters {
    pub top_k: usize,
    pub top_k_simd: usize,
    pub top_p: f16,
    pub min_p: f16,
    pub do_sample: bool,
    pub eos_token_id_list: Vec<usize>,
}

pub struct ThreadingConfig {
    pub worker_threads: usize,
    pub async_threads: usize,
    pub total_threads: usize,
}

pub fn load_model_config(
    model_dir: &str,
) -> Result<(Config, Option<GenerationConfig>, String), String> {
    let config = Config::load_from_file(format!("{}/config.json", model_dir))
        .map_err(|e| format!("failed to load config: {}", e))?;
    let generation_config =
        GenerationConfig::load_from_file(format!("{}/generation_config.json", model_dir)).ok();

    if let Some(gen_cfg) = &generation_config {
        println!("Loaded generation config: {:?}", gen_cfg);
    }

    Ok((config, generation_config, model_dir.to_string()))
}

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

pub fn load_model_parameters(model_dir: &str) -> Result<(), String> {
    let params = crate::runtime::SafeTensorsLoader::new(model_dir)
        .and_then(|loader| loader.load_all_weights_f16())
        .map_err(|e| format!("failed to load model parameters: {}", e))?;

    println!("Loaded {} parameter tensors", params.len());
    f16::init_global_strict(params);
    Ok(())
}

pub fn determine_thread_config(generation_config: &Option<GenerationConfig>) -> ThreadingConfig {
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
    let total_threads = core_ids.len().max(1).min(requested_thread_num);
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
