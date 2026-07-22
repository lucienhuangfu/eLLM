use crate::config::ResolvedConfig;
use crate::runtime::session::SessionMode;

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
