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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(alias = "model-path")]
    pub model: String,
    #[serde(default = "default_min_p")]
    #[serde(alias = "min-p")]
    pub min_p: f64,
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
    #[serde(alias = "tokenizer-revision")]
    pub tokenizer_revision: Option<String>,
    #[serde(default)]
    #[serde(alias = "download-dir")]
    pub download_dir: Option<String>,
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
    #[serde(default)]
    #[serde(alias = "prefill-chunk-size")]
    pub prefill_chunk_size: Option<usize>,
    #[serde(default = "default_scheduling_policy")]
    #[serde(alias = "scheduling-policy")]
    pub scheduling_policy: SchedulingPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_command")]
    pub command: Command,
    pub model: ModelConfig,
    #[serde(default)]
    pub scheduler: SchedulerConfig,
    #[serde(default)]
    pub serve: Option<ServeConfig>,
    #[serde(default)]
    pub chat: Option<ChatConfig>,
}

#[derive(Debug, Clone)]
pub struct ResolvedModelConfig {
    pub raw: ModelConfig,
    pub served_model_name: String,
    pub effective_tokenizer: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub command: Command,
    pub model: ResolvedModelConfig,
    pub scheduler: SchedulerConfig,
    pub serve: Option<ServeConfig>,
    pub chat: Option<ChatConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeConfig {
    #[serde(default = "default_host")]
    #[serde(alias = "host")]
    pub host: String,
    #[serde(default = "default_port")]
    #[serde(alias = "port")]
    pub port: u16,
    #[serde(default)]
    #[serde(alias = "log-requests")]
    pub log_requests: bool,
    #[serde(default)]
    #[serde(alias = "api-key")]
    pub api_key: Option<String>,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            log_requests: false,
            api_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    #[serde(default)]
    #[serde(alias = "system-prompt")]
    pub system_prompt: Option<String>,
    #[serde(default)]
    #[serde(alias = "stream")]
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
            prefill_chunk_size: None,
            scheduling_policy: default_scheduling_policy(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            min_p: default_min_p(),
            tokenizer: None,
            dtype: default_dtype(),
            max_model_len: None,
            trust_remote_code: false,
            quantization: None,
            kv_cache_dtype: None,
            served_model_name: None,
            revision: None,
            tokenizer_revision: None,
            download_dir: None,
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
    ModelDtype::Fp16
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

pub(crate) fn infer_served_model_name(model: &str) -> String {
    let trimmed = model.trim().trim_end_matches('/');
    let candidate = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed).trim();
    if candidate.is_empty() {
        "model".to_string()
    } else {
        candidate.to_string()
    }
}
