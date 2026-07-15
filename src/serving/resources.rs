use std::sync::Arc;

use crate::config::ResolvedConfig;
use crate::mem_mgr::allocator::AlignedBox;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{SessionMode, SlotManager};
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::{initialize_runtime, RuntimeContext};

use super::parser::{ParserOptions, ParserRule};

// ── Serving-layer configuration ───────────────────────────────────────────────

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

// ── Serving-layer runtime resources ──────────────────────────────────────────

/// All handles needed by the HTTP serving layer.
///
/// Contains all inference-runtime handles from [`RuntimeContext`] plus
/// serving-specific fields (parser options, thread counts).
pub struct ServingResources<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    pub batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    pub scheduler: Arc<Scheduler>,
    pub parser_options: ParserOptions,
    pub api_threads: usize,
    pub blocking_threads: usize,
    pub _sequences_box: AlignedBox<usize>,
    pub session_mode: SessionMode,
    pub slot_reuse_timeout_ms: usize,
    pub slot_manager: Arc<SlotManager<T>>,
}

impl<T> From<(RuntimeContext<T>, ParserOptions)> for ServingResources<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    fn from((ctx, parser_options): (RuntimeContext<T>, ParserOptions)) -> Self {
        let api_threads = ctx.thread_config.api_threads;
        let blocking_threads = ctx.thread_config.blocking_threads;
        ServingResources {
            batch_sequences: ctx.batch_sequences,
            scheduler: ctx.scheduler,
            parser_options,
            api_threads,
            blocking_threads,
            _sequences_box: ctx._sequences_box,
            session_mode: ctx.session_mode,
            slot_reuse_timeout_ms: ctx.slot_reuse_timeout_ms,
            slot_manager: ctx.slot_manager,
        }
    }
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

/// Bootstrap the serving layer from a `ResolvedConfig`.
///
/// Delegates all inference-runtime setup (weight loading, model warm-up,
/// executor pool) to [`runtime::initialize_runtime`], then wraps the
/// result with serving-specific configuration.
pub fn initialize_serving_resources(
    resolved_config: &ResolvedConfig,
) -> Result<ServingResources<f16>, Box<dyn std::error::Error>> {
    let config = ServingConfig::from_resolved_config(resolved_config);

    let model_config = crate::transformer::config::Config::load_from_file(format!(
        "{}/config.json",
        config.model_dir
    ))
    .map_err(|e| format!("failed to load config: {}", e))?;

    let parser_options = ParserOptions {
        rule: ParserRule::for_model_family(&model_config.family),
        reasoning_parser: config.reasoning_parser_enabled,
        tool_call_parser: config.tool_call_parser_enabled,
    };

    let ctx = initialize_runtime(
        resolved_config,
        config.api_server_count,
        config.batch_size,
        config.sequence_length,
        config.chunk_size,
        config.session_mode,
        config.slot_reuse_timeout_ms,
    )?;

    Ok(ServingResources::from((ctx, parser_options)))
}
