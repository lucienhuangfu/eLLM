use crate::config::ResolvedConfig;
use crate::runtime::{initialize_runtime, RuntimeContext};

use crate::serving::config::ServingConfig;
use crate::serving::parser::ParserOptions;

pub fn initialize_serving_resources(
    resolved_config: &ResolvedConfig,
) -> Result<(RuntimeContext<f16>, ParserOptions), Box<dyn std::error::Error>> {
    let config = ServingConfig::from_resolved_config(resolved_config);

    let model_config = crate::transformer::config::Config::load_from_file(format!(
        "{}/config.json",
        config.model_dir
    ))
    .map_err(|e| format!("failed to load config: {}", e))?;

    let parser_options = config.build_parser_options(&model_config);

    let ctx = initialize_runtime(
        resolved_config,
        config.api_server_count,
        config.batch_size,
        config.sequence_length,
        config.chunk_size,
        config.session_mode,
        config.slot_reuse_timeout_ms,
    )?;

    Ok((ctx, parser_options))
}
