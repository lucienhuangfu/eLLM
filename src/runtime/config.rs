use crate::config::GenerationConfig;
use crate::model_family::config::Config;

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
