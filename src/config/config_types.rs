use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    Serve,
    Chat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ModelDtype {
    Auto,
    Fp16,
    Bf16,
    Fp32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingPolicy {
    Fair,
    Fifo,
    Priority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerMode {
    Auto,
    Hf,
    Slow,
    Mistral,
    DeepseekV32,
    DeepseekV4,
    QwenVl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum RunnerType {
    Auto,
    Draft,
    Generate,
    Pooling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ConvertType {
    Auto,
    Classify,
    Embed,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(alias = "model-path")]
    pub model: String,

    #[serde(default = "default_tokenizer_mode")]
    #[serde(alias = "tokenizer-mode")]
    pub tokenizer_mode: TokenizerMode,

    #[serde(default)]
    #[serde(alias = "tokenizer-path")]
    pub tokenizer: Option<String>,

    #[serde(default = "default_dtype")]
    pub dtype: ModelDtype,

    #[serde(default)]
    #[serde(alias = "max-model-len")]
    pub max_model_len: Option<usize>,

    #[serde(default)]
    #[serde(alias = "trust-remote-code")]
    pub trust_remote_code: bool,

    #[serde(default)]
    pub quantization: Option<String>,

    #[serde(default)]
    #[serde(alias = "kv-cache-dtype")]
    pub kv_cache_dtype: Option<String>,

    #[serde(default)]
    #[serde(alias = "served-model-name")]
    pub served_model_name: Option<String>,

    #[serde(default)]
    pub revision: Option<String>,

    #[serde(default)]
    #[serde(alias = "code-revision")]
    pub code_revision: Option<String>,

    #[serde(default)]
    #[serde(alias = "tokenizer-revision")]
    pub tokenizer_revision: Option<String>,

    #[serde(default)]
    #[serde(alias = "download-dir")]
    pub download_dir: Option<String>,

    #[serde(default = "default_seed")]
    pub seed: usize,

    #[serde(default)]
    #[serde(alias = "hf-config-path")]
    pub hf_config_path: Option<String>,

    #[serde(default)]
    #[serde(alias = "allowed-local-media-path")]
    pub allowed_local_media_path: String,

    #[serde(default)]
    #[serde(alias = "allowed-media-domains")]
    pub allowed_media_domains: Vec<String>,

    #[serde(default = "default_max_logprobs")]
    #[serde(alias = "max-logprobs")]
    pub max_logprobs: usize,

    #[serde(default)]
    #[serde(alias = "disable-sliding-window")]
    pub disable_sliding_window: bool,

    #[serde(default)]
    #[serde(alias = "disable-cascade-attn")]
    pub disable_cascade_attn: bool,

    #[serde(default = "default_min_p")]
    #[serde(alias = "min-p")]
    pub min_p: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_max_num_seqs")]
    #[serde(alias = "max-num-seqs")]
    pub max_num_seqs: usize,

    #[serde(default = "default_max_num_batched_tokens")]
    #[serde(alias = "max-num-batched-tokens")]
    pub max_num_batched_tokens: usize,

    #[serde(default = "default_enable_continuous_batching")]
    #[serde(alias = "enable-continuous-batching")]
    pub enable_continuous_batching: bool,

    #[serde(default = "default_scheduling_policy")]
    #[serde(alias = "scheduling-policy")]
    pub scheduling_policy: SchedulingPolicy,

    #[serde(default = "default_dialogue_cache_enabled")]
    #[serde(alias = "dialogue-cache-enabled")]
    pub dialogue_cache_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    #[serde(default = "default_runner")]
    pub runner: RunnerType,

    #[serde(default = "default_convert")]
    pub convert: ConvertType,

    #[serde(default)]
    #[serde(alias = "enforce-eager")]
    pub enforce_eager: bool,

    #[serde(default)]
    #[serde(alias = "enable-return-routed-experts")]
    pub enable_return_routed_experts: bool,

    #[serde(default)]
    #[serde(alias = "use-fp64-gumbel")]
    pub use_fp64_gumbel: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_command")]
    pub command: Command,

    pub model: ModelConfig,

    #[serde(default)]
    pub scheduler: SchedulerConfig,

    #[serde(default)]
    pub engine: EngineConfig,

    #[serde(default)]
    pub serve: Option<ServeConfig>,

    #[serde(default)]
    pub chat: Option<ChatConfig>,
}

#[derive(Debug, Clone)]
pub struct ResolvedModelConfig {
    pub raw_config: ModelConfig,
    pub served_model_name: String,
    pub effective_tokenizer: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub command: Command,
    pub model: ResolvedModelConfig,
    pub scheduler: SchedulerConfig,
    pub engine: EngineConfig,
    pub serve: Option<ServeConfig>,
    pub chat: Option<ChatConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeConfig {
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default)]
    #[serde(alias = "log-requests")]
    pub log_requests: bool,

    #[serde(default)]
    #[serde(alias = "api-key")]
    pub api_key: Option<String>,

    #[serde(default = "default_reasoning_parser_enabled")]
    #[serde(alias = "reasoning-parser-enabled")]
    pub reasoning_parser_enabled: bool,

    #[serde(default = "default_tool_call_parser_enabled")]
    #[serde(alias = "tool-call-parser-enabled")]
    pub tool_call_parser_enabled: bool,

    #[serde(default = "default_api_server_count")]
    #[serde(alias = "api-server-count")]
    pub api_server_count: usize,

    #[serde(default)]
    #[serde(alias = "uds")]
    pub uds: Option<String>,

    #[serde(default)]
    #[serde(alias = "ssl-keyfile")]
    pub ssl_keyfile: Option<String>,

    #[serde(default)]
    #[serde(alias = "ssl-certfile")]
    pub ssl_certfile: Option<String>,

    #[serde(default)]
    #[serde(alias = "ssl-ca-certs")]
    pub ssl_ca_certs: Option<String>,

    #[serde(default)]
    #[serde(alias = "allow-credentials")]
    pub allow_credentials: bool,

    #[serde(default = "default_allowed_origins")]
    #[serde(alias = "allowed-origins")]
    pub allowed_origins: Vec<String>,

    #[serde(default = "default_allowed_methods")]
    #[serde(alias = "allowed-methods")]
    pub allowed_methods: Vec<String>,

    #[serde(default = "default_allowed_headers")]
    #[serde(alias = "allowed-headers")]
    pub allowed_headers: Vec<String>,

    #[serde(default = "default_slot_reuse_timeout_ms")]
    #[serde(alias = "slot-reuse-timeout-ms")]
    pub slot_reuse_timeout_ms: usize,

    #[serde(default)]
    #[serde(alias = "max-slot-size")]
    pub max_slot_size: Option<usize>,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            log_requests: false,
            api_key: None,
            reasoning_parser_enabled: default_reasoning_parser_enabled(),
            tool_call_parser_enabled: default_tool_call_parser_enabled(),
            api_server_count: default_api_server_count(),
            uds: None,
            ssl_keyfile: None,
            ssl_certfile: None,
            ssl_ca_certs: None,
            allow_credentials: false,
            allowed_origins: default_allowed_origins(),
            allowed_methods: default_allowed_methods(),
            allowed_headers: default_allowed_headers(),
            slot_reuse_timeout_ms: default_slot_reuse_timeout_ms(),
            max_slot_size: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    #[serde(default)]
    #[serde(alias = "system-prompt")]
    pub system_prompt: Option<String>,

    #[serde(default)]
    pub stream: bool,

    #[serde(default)]
    #[serde(alias = "max-turns")]
    pub max_turns: Option<usize>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            stream: false,
            max_turns: None,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: default_max_num_seqs(),
            max_num_batched_tokens: default_max_num_batched_tokens(),
            enable_continuous_batching: default_enable_continuous_batching(),
            scheduling_policy: default_scheduling_policy(),
            dialogue_cache_enabled: default_dialogue_cache_enabled(),
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            runner: default_runner(),
            convert: default_convert(),
            enforce_eager: false,
            enable_return_routed_experts: false,
            use_fp64_gumbel: false,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            tokenizer_mode: default_tokenizer_mode(),
            tokenizer: None,
            dtype: default_dtype(),
            max_model_len: None,
            trust_remote_code: false,
            quantization: None,
            kv_cache_dtype: None,
            served_model_name: None,
            revision: None,
            code_revision: None,
            tokenizer_revision: None,
            download_dir: None,
            seed: default_seed(),
            hf_config_path: None,
            allowed_local_media_path: String::new(),
            allowed_media_domains: Vec::new(),
            max_logprobs: default_max_logprobs(),
            disable_sliding_window: false,
            disable_cascade_attn: true,
            min_p: default_min_p(),
        }
    }
}

pub(crate) fn default_command() -> Command {
    Command::Serve
}

pub(crate) fn default_min_p() -> f64 {
    0.0
}

pub(crate) fn default_dtype() -> ModelDtype {
    ModelDtype::Auto
}

pub(crate) fn default_tokenizer_mode() -> TokenizerMode {
    TokenizerMode::Auto
}

pub(crate) fn default_runner() -> RunnerType {
    RunnerType::Auto
}

pub(crate) fn default_convert() -> ConvertType {
    ConvertType::Auto
}

pub(crate) fn default_max_num_seqs() -> usize {
    256
}

pub(crate) fn default_max_num_batched_tokens() -> usize {
    8192
}

pub(crate) fn default_enable_continuous_batching() -> bool {
    true
}

pub(crate) fn default_scheduling_policy() -> SchedulingPolicy {
    SchedulingPolicy::Fair
}

pub(crate) fn default_host() -> String {
    "127.0.0.1".to_string()
}

pub(crate) fn default_port() -> u16 {
    8000
}

pub(crate) fn default_reasoning_parser_enabled() -> bool {
    true
}

pub(crate) fn default_tool_call_parser_enabled() -> bool {
    true
}

pub(crate) fn default_dialogue_cache_enabled() -> bool {
    false
}

pub(crate) fn default_api_server_count() -> usize {
    2
}

pub(crate) fn default_seed() -> usize {
    0
}

pub(crate) fn default_max_logprobs() -> usize {
    20
}

pub(crate) fn default_allowed_origins() -> Vec<String> {
    vec!["*".to_string()]
}

pub(crate) fn default_allowed_methods() -> Vec<String> {
    vec!["*".to_string()]
}

pub(crate) fn default_allowed_headers() -> Vec<String> {
    vec!["*".to_string()]
}

pub(crate) fn default_slot_reuse_timeout_ms() -> usize {
    30000
}

pub(crate) fn infer_served_model_name(model: &str) -> String {
    let trimmed = model.trim().trim_end_matches('/');
    let candidate = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed).trim();
    if candidate.is_empty() {
        "model".to_string()
    } else {
        candidate.to_string()
    }
}
