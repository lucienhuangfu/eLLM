use super::config_types::{
    ChatConfig, Command, Config, ConvertType, EngineConfig, ModelDtype, RunnerType,
    SchedulingPolicy, ServeConfig, TokenizerMode,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Clone, Parser)]
#[command(name = "ellm")]
#[command(about = "Simplified LLM runtime configuration CLI")]
#[command(help_template = "\
{name} {version}
{about}

Usage: {usage}

Commands:
{commands}

Arguments:
{arguments}

Options:
{options}

For JSON CLI arguments, use:
  --json-arg '{\"key\": \"value\"}'
  --json-arg.key value
")]
pub struct Cli {
    #[arg(
        long = "config",
        value_name = "FILE",
        help = "Read CLI options from a config file. Must be a YAML file with options matching vLLM's serve_args format."
    )]
    pub config: Option<PathBuf>,

    #[arg(
        long = "json-arg",
        value_name = "JSON",
        help = "JSON-formatted arguments"
    )]
    pub json_arg: Vec<String>,

    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub enum CliCommand {
    Serve(ServeArgs),
    Chat(ChatArgs),
}

#[derive(Debug, Clone, Default, Args)]
pub struct SharedArgs {
    #[arg(
        short = 'p',
        long = "min-p",
        default_value = "0.0",
        help = "Minimum probability for nucleus sampling"
    )]
    pub min_p: Option<f64>,

    #[arg(short = 'm', long = "model", help = "Model name or path")]
    pub model: Option<String>,

    #[arg(long = "tokenizer", help = "Tokenizer name or path")]
    pub tokenizer: Option<String>,

    #[arg(long = "tokenizer-mode", value_enum, help = "Tokenizer mode")]
    pub tokenizer_mode: Option<TokenizerMode>,

    #[arg(short = 'd', long = "dtype", value_enum, help = "Model data type")]
    pub dtype: Option<ModelDtype>,

    #[arg(long = "max-model-len", help = "Maximum model context length")]
    pub max_model_len: Option<usize>,

    #[arg(long = "trust-remote-code", num_args = 0..=1, default_missing_value = "true", help = "Trust remote code")]
    pub trust_remote_code: Option<bool>,

    #[arg(short = 'q', long = "quantization", help = "Quantization method")]
    pub quantization: Option<String>,

    #[arg(long = "kv-cache-dtype", help = "KV cache data type")]
    pub kv_cache_dtype: Option<String>,

    #[arg(long = "served-model-name", help = "Served model name")]
    pub served_model_name: Option<String>,

    #[arg(long = "revision", help = "Model revision")]
    pub revision: Option<String>,

    #[arg(long = "code-revision", help = "Code revision")]
    pub code_revision: Option<String>,

    #[arg(long = "tokenizer-revision", help = "Tokenizer revision")]
    pub tokenizer_revision: Option<String>,

    #[arg(long = "download-dir", help = "Download directory")]
    pub download_dir: Option<String>,

    #[arg(short = 's', long = "seed", default_value = "0", help = "Random seed")]
    pub seed: Option<usize>,

    #[arg(long = "hf-config-path", help = "Hugging Face config path")]
    pub hf_config_path: Option<String>,

    #[arg(long = "max-logprobs", default_value = "20", help = "Maximum logprobs")]
    pub max_logprobs: Option<usize>,

    #[arg(long = "disable-sliding-window", num_args = 0..=1, default_missing_value = "true", help = "Disable sliding window")]
    pub disable_sliding_window: Option<bool>,

    #[arg(long = "disable-cascade-attn", num_args = 0..=1, default_missing_value = "true", help = "Disable cascade attention")]
    pub disable_cascade_attn: Option<bool>,

    #[arg(long = "runner", value_enum, help = "Runner type")]
    pub runner: Option<RunnerType>,

    #[arg(long = "convert", value_enum, help = "Convert type")]
    pub convert: Option<ConvertType>,

    #[arg(long = "enforce-eager", num_args = 0..=1, default_missing_value = "true", help = "Enforce eager mode")]
    pub enforce_eager: Option<bool>,

    #[arg(long = "enable-return-routed-experts", num_args = 0..=1, default_missing_value = "true", help = "Return routed experts")]
    pub enable_return_routed_experts: Option<bool>,

    #[arg(long = "use-fp64-gumbel", num_args = 0..=1, default_missing_value = "true", help = "Use FP64 Gumbel")]
    pub use_fp64_gumbel: Option<bool>,

    #[arg(long = "max-num-seqs", help = "Maximum number of sequences")]
    pub max_num_seqs: Option<usize>,

    #[arg(
        long = "max-num-batched-tokens",
        help = "Maximum number of batched tokens"
    )]
    pub max_num_batched_tokens: Option<usize>,

    #[arg(long = "enable-continuous-batching", num_args = 0..=1, default_missing_value = "true", help = "Enable continuous batching")]
    pub enable_continuous_batching: Option<bool>,

    #[arg(long = "scheduling-policy", value_enum, help = "Scheduling policy")]
    pub scheduling_policy: Option<SchedulingPolicy>,

    #[arg(long = "schedule-timeout-ms", help = "Schedule timeout in ms")]
    pub schedule_timeout_ms: Option<usize>,

    #[arg(long = "dialogue-cache-enabled", num_args = 0..=1, default_missing_value = "true", help = "Enable dialogue cache")]
    pub dialogue_cache_enabled: Option<bool>,
}

#[derive(Debug, Clone, Args)]
pub struct ServeArgs {
    #[command(flatten)]
    pub shared: SharedArgs,

    #[arg(long = "config", help = "Config file path")]
    pub config: Option<PathBuf>,

    #[arg(short = 'H', long = "host", help = "Host address")]
    pub host: Option<String>,

    #[arg(short = 'P', long = "port", help = "Port number")]
    pub port: Option<u16>,

    #[arg(long = "log-requests", num_args = 0..=1, default_missing_value = "true", help = "Log requests")]
    pub log_requests: Option<bool>,

    #[arg(long = "api-key", help = "API key")]
    pub api_key: Option<String>,

    #[arg(long = "reasoning-parser-enabled", num_args = 0..=1, default_missing_value = "true", help = "Enable reasoning parser")]
    pub reasoning_parser_enabled: Option<bool>,

    #[arg(long = "tool-call-parser-enabled", num_args = 0..=1, default_missing_value = "true", help = "Enable tool call parser")]
    pub tool_call_parser_enabled: Option<bool>,

    #[arg(long = "api-server-count", help = "Number of API server threads")]
    pub api_server_count: Option<usize>,

    #[arg(long = "uds", help = "Unix domain socket path")]
    pub uds: Option<String>,

    #[arg(long = "ssl-keyfile", help = "SSL key file")]
    pub ssl_keyfile: Option<String>,

    #[arg(long = "ssl-certfile", help = "SSL cert file")]
    pub ssl_certfile: Option<String>,

    #[arg(long = "ssl-ca-certs", help = "SSL CA certs")]
    pub ssl_ca_certs: Option<String>,

    #[arg(long = "allow-credentials", num_args = 0..=1, default_missing_value = "true", help = "Allow credentials")]
    pub allow_credentials: Option<bool>,

    #[arg(
        long = "allowed-origins",
        value_delimiter = ',',
        help = "Allowed origins"
    )]
    pub allowed_origins: Option<Vec<String>>,

    #[arg(
        long = "allowed-methods",
        value_delimiter = ',',
        help = "Allowed methods"
    )]
    pub allowed_methods: Option<Vec<String>>,

    #[arg(
        long = "allowed-headers",
        value_delimiter = ',',
        help = "Allowed headers"
    )]
    pub allowed_headers: Option<Vec<String>>,
}

#[derive(Debug, Clone, Args)]
pub struct ChatArgs {
    #[command(flatten)]
    pub shared: SharedArgs,

    #[arg(long = "config", help = "Config file path")]
    pub config: Option<PathBuf>,

    #[arg(long = "system-prompt", help = "System prompt")]
    pub system_prompt: Option<String>,

    #[arg(short = 'S', long = "stream", num_args = 0..=1, default_missing_value = "true", help = "Stream output")]
    pub stream: Option<bool>,

    #[arg(long = "max-turns", help = "Maximum turns")]
    pub max_turns: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct VllmConfigFile {
    #[serde(flatten)]
    pub model: Option<VllmModelConfig>,

    #[serde(flatten)]
    pub scheduler: Option<VllmSchedulerConfig>,

    #[serde(flatten)]
    pub engine: Option<VllmEngineConfig>,

    #[serde(flatten)]
    pub server: Option<VllmServerConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct VllmModelConfig {
    pub model: Option<String>,
    pub tokenizer: Option<String>,
    #[serde(alias = "tokenizer_mode")]
    pub tokenizer_mode: Option<String>,
    pub dtype: Option<String>,
    #[serde(alias = "max_model_len")]
    pub max_model_len: Option<usize>,
    #[serde(alias = "trust_remote_code")]
    pub trust_remote_code: Option<bool>,
    pub quantization: Option<String>,
    #[serde(alias = "kv_cache_dtype")]
    pub kv_cache_dtype: Option<String>,
    #[serde(alias = "served_model_name")]
    pub served_model_name: Option<String>,
    pub revision: Option<String>,
    #[serde(alias = "code_revision")]
    pub code_revision: Option<String>,
    #[serde(alias = "tokenizer_revision")]
    pub tokenizer_revision: Option<String>,
    #[serde(alias = "download_dir")]
    pub download_dir: Option<String>,
    pub seed: Option<usize>,
    #[serde(alias = "hf_config_path")]
    pub hf_config_path: Option<String>,
    #[serde(alias = "max_logprobs")]
    pub max_logprobs: Option<usize>,
    #[serde(alias = "disable_sliding_window")]
    pub disable_sliding_window: Option<bool>,
    #[serde(alias = "disable_cascade_attn")]
    pub disable_cascade_attn: Option<bool>,
    #[serde(alias = "min_p")]
    pub min_p: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct VllmSchedulerConfig {
    #[serde(alias = "max_num_seqs")]
    pub max_num_seqs: Option<usize>,
    #[serde(alias = "max_num_batched_tokens")]
    pub max_num_batched_tokens: Option<usize>,
    #[serde(alias = "enable_continuous_batching")]
    pub enable_continuous_batching: Option<bool>,
    #[serde(alias = "scheduling_policy")]
    pub scheduling_policy: Option<String>,
    #[serde(alias = "schedule_timeout_ms")]
    pub schedule_timeout_ms: Option<usize>,
    #[serde(alias = "dialogue_cache_enabled")]
    pub dialogue_cache_enabled: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct VllmEngineConfig {
    pub runner: Option<String>,
    pub convert: Option<String>,
    #[serde(alias = "enforce_eager")]
    pub enforce_eager: Option<bool>,
    #[serde(alias = "enable_return_routed_experts")]
    pub enable_return_routed_experts: Option<bool>,
    #[serde(alias = "use_fp64_gumbel")]
    pub use_fp64_gumbel: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct VllmServerConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    #[serde(alias = "log_requests")]
    pub log_requests: Option<bool>,
    #[serde(alias = "api_key")]
    pub api_key: Option<String>,
    #[serde(alias = "reasoning_parser_enabled")]
    pub reasoning_parser_enabled: Option<bool>,
    #[serde(alias = "tool_call_parser_enabled")]
    pub tool_call_parser_enabled: Option<bool>,
    #[serde(alias = "api_server_count")]
    pub api_server_count: Option<usize>,
    pub uds: Option<String>,
    #[serde(alias = "ssl_keyfile")]
    pub ssl_keyfile: Option<String>,
    #[serde(alias = "ssl_certfile")]
    pub ssl_certfile: Option<String>,
    #[serde(alias = "ssl_ca_certs")]
    pub ssl_ca_certs: Option<String>,
    #[serde(alias = "allow_credentials")]
    pub allow_credentials: Option<bool>,
    #[serde(alias = "allowed_origins")]
    pub allowed_origins: Option<Vec<String>>,
    #[serde(alias = "allowed_methods")]
    pub allowed_methods: Option<Vec<String>>,
    #[serde(alias = "allowed_headers")]
    pub allowed_headers: Option<Vec<String>>,
}

impl VllmConfigFile {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: VllmConfigFile = serde_yaml::from_reader(reader)?;
        Ok(config)
    }

    pub fn apply_to_config(&self, config: &mut Config) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(model) = &self.model {
            self.apply_model_config(model, config)?;
        }
        if let Some(scheduler) = &self.scheduler {
            self.apply_scheduler_config(scheduler, config)?;
        }
        if let Some(engine) = &self.engine {
            self.apply_engine_config(engine, config)?;
        }
        if let Some(server) = &self.server {
            self.apply_server_config(server, config)?;
        }
        Ok(())
    }

    fn apply_model_config(
        &self,
        model: &VllmModelConfig,
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(model_path) = &model.model {
            config.model.model = model_path.clone();
        }
        if let Some(tokenizer) = &model.tokenizer {
            config.model.tokenizer = Some(tokenizer.clone());
        }
        if let Some(tokenizer_mode) = &model.tokenizer_mode {
            config.model.tokenizer_mode = TokenizerMode::from_str(tokenizer_mode, false)
                .map_err(|_| format!("Invalid tokenizer_mode: {}", tokenizer_mode))?;
        }
        if let Some(dtype) = &model.dtype {
            config.model.dtype = ModelDtype::from_str(dtype, false)
                .map_err(|_| format!("Invalid dtype: {}", dtype))?;
        }
        if let Some(max_model_len) = model.max_model_len {
            config.model.max_model_len = Some(max_model_len);
        }
        if let Some(trust_remote_code) = model.trust_remote_code {
            config.model.trust_remote_code = trust_remote_code;
        }
        if let Some(quantization) = &model.quantization {
            config.model.quantization = Some(quantization.clone());
        }
        if let Some(kv_cache_dtype) = &model.kv_cache_dtype {
            config.model.kv_cache_dtype = Some(kv_cache_dtype.clone());
        }
        if let Some(served_model_name) = &model.served_model_name {
            config.model.served_model_name = Some(served_model_name.clone());
        }
        if let Some(revision) = &model.revision {
            config.model.revision = Some(revision.clone());
        }
        if let Some(code_revision) = &model.code_revision {
            config.model.code_revision = Some(code_revision.clone());
        }
        if let Some(tokenizer_revision) = &model.tokenizer_revision {
            config.model.tokenizer_revision = Some(tokenizer_revision.clone());
        }
        if let Some(download_dir) = &model.download_dir {
            config.model.download_dir = Some(download_dir.clone());
        }
        if let Some(seed) = model.seed {
            config.model.seed = seed;
        }
        if let Some(hf_config_path) = &model.hf_config_path {
            config.model.hf_config_path = Some(hf_config_path.clone());
        }
        if let Some(max_logprobs) = model.max_logprobs {
            config.model.max_logprobs = max_logprobs;
        }
        if let Some(disable_sliding_window) = model.disable_sliding_window {
            config.model.disable_sliding_window = disable_sliding_window;
        }
        if let Some(disable_cascade_attn) = model.disable_cascade_attn {
            config.model.disable_cascade_attn = disable_cascade_attn;
        }
        if let Some(min_p) = model.min_p {
            config.model.min_p = min_p;
        }
        Ok(())
    }

    fn apply_scheduler_config(
        &self,
        scheduler: &VllmSchedulerConfig,
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(max_num_seqs) = scheduler.max_num_seqs {
            config.scheduler.max_num_seqs = max_num_seqs;
        }
        if let Some(max_num_batched_tokens) = scheduler.max_num_batched_tokens {
            config.scheduler.max_num_batched_tokens = max_num_batched_tokens;
        }
        if let Some(enable_continuous_batching) = scheduler.enable_continuous_batching {
            config.scheduler.enable_continuous_batching = enable_continuous_batching;
        }
        if let Some(scheduling_policy) = &scheduler.scheduling_policy {
            config.scheduler.scheduling_policy =
                SchedulingPolicy::from_str(scheduling_policy, false)
                    .map_err(|_| format!("Invalid scheduling_policy: {}", scheduling_policy))?;
        }
        if let Some(schedule_timeout_ms) = scheduler.schedule_timeout_ms {
            config.scheduler.schedule_timeout_ms = schedule_timeout_ms;
        }
        if let Some(dialogue_cache_enabled) = scheduler.dialogue_cache_enabled {
            config.scheduler.dialogue_cache_enabled = dialogue_cache_enabled;
        }
        Ok(())
    }

    fn apply_engine_config(
        &self,
        engine: &VllmEngineConfig,
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(runner) = &engine.runner {
            config.engine.runner = RunnerType::from_str(runner, false)
                .map_err(|_| format!("Invalid runner: {}", runner))?;
        }
        if let Some(convert) = &engine.convert {
            config.engine.convert = ConvertType::from_str(convert, false)
                .map_err(|_| format!("Invalid convert: {}", convert))?;
        }
        if let Some(enforce_eager) = engine.enforce_eager {
            config.engine.enforce_eager = enforce_eager;
        }
        if let Some(enable_return_routed_experts) = engine.enable_return_routed_experts {
            config.engine.enable_return_routed_experts = enable_return_routed_experts;
        }
        if let Some(use_fp64_gumbel) = engine.use_fp64_gumbel {
            config.engine.use_fp64_gumbel = use_fp64_gumbel;
        }
        Ok(())
    }

    fn apply_server_config(
        &self,
        server: &VllmServerConfig,
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if config.serve.is_none() {
            config.serve = Some(ServeConfig::default());
        }
        if let Some(serve) = &mut config.serve {
            if let Some(host) = &server.host {
                serve.host = host.clone();
            }
            if let Some(port) = server.port {
                serve.port = port;
            }
            if let Some(log_requests) = server.log_requests {
                serve.log_requests = log_requests;
            }
            if let Some(api_key) = &server.api_key {
                serve.api_key = Some(api_key.clone());
            }
            if let Some(reasoning_parser_enabled) = server.reasoning_parser_enabled {
                serve.reasoning_parser_enabled = reasoning_parser_enabled;
            }
            if let Some(tool_call_parser_enabled) = server.tool_call_parser_enabled {
                serve.tool_call_parser_enabled = tool_call_parser_enabled;
            }
            if let Some(api_server_count) = server.api_server_count {
                serve.api_server_count = api_server_count;
            }
            if let Some(uds) = &server.uds {
                serve.uds = Some(uds.clone());
            }
            if let Some(ssl_keyfile) = &server.ssl_keyfile {
                serve.ssl_keyfile = Some(ssl_keyfile.clone());
            }
            if let Some(ssl_certfile) = &server.ssl_certfile {
                serve.ssl_certfile = Some(ssl_certfile.clone());
            }
            if let Some(ssl_ca_certs) = &server.ssl_ca_certs {
                serve.ssl_ca_certs = Some(ssl_ca_certs.clone());
            }
            if let Some(allow_credentials) = server.allow_credentials {
                serve.allow_credentials = allow_credentials;
            }
            if let Some(allowed_origins) = &server.allowed_origins {
                serve.allowed_origins = allowed_origins.clone();
            }
            if let Some(allowed_methods) = &server.allowed_methods {
                serve.allowed_methods = allowed_methods.clone();
            }
            if let Some(allowed_headers) = &server.allowed_headers {
                serve.allowed_headers = allowed_headers.clone();
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct JsonArgs {
    pub model: Option<HashMap<String, Value>>,
    pub scheduler: Option<HashMap<String, Value>>,
    pub engine: Option<HashMap<String, Value>>,
    pub serve: Option<HashMap<String, Value>>,
    pub chat: Option<HashMap<String, Value>>,
}

impl JsonArgs {
    pub fn parse(json_args: &[String]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut model = HashMap::new();
        let mut scheduler = HashMap::new();
        let mut engine = HashMap::new();
        let mut serve = HashMap::new();
        let mut chat = HashMap::new();

        for arg in json_args {
            if arg.starts_with('{') {
                let value: Value = serde_json::from_str(arg)?;
                if let Value::Object(map) = value {
                    Self::merge_json(
                        &mut model,
                        &mut scheduler,
                        &mut engine,
                        &mut serve,
                        &mut chat,
                        &map,
                    );
                }
            } else if let Some((key, value)) = arg.split_once('=') {
                let key = key.trim();
                let value = value.trim();

                let value: Value = if value.starts_with('{') || value.starts_with('[') {
                    serde_json::from_str(value)?
                } else if value.eq_ignore_ascii_case("true") {
                    Value::Bool(true)
                } else if value.eq_ignore_ascii_case("false") {
                    Value::Bool(false)
                } else if let Ok(n) = value.parse::<i64>() {
                    Value::Number(n.into())
                } else if let Ok(n) = value.parse::<f64>() {
                    Value::Number(
                        serde_json::Number::from_f64(n).unwrap_or(serde_json::Number::from(0)),
                    )
                } else {
                    Value::String(value.to_string())
                };

                Self::set_nested_key(
                    key,
                    value,
                    &mut model,
                    &mut scheduler,
                    &mut engine,
                    &mut serve,
                    &mut chat,
                );
            }
        }

        Ok(Self {
            model: if model.is_empty() { None } else { Some(model) },
            scheduler: if scheduler.is_empty() {
                None
            } else {
                Some(scheduler)
            },
            engine: if engine.is_empty() {
                None
            } else {
                Some(engine)
            },
            serve: if serve.is_empty() { None } else { Some(serve) },
            chat: if chat.is_empty() { None } else { Some(chat) },
        })
    }

    fn merge_json(
        model: &mut HashMap<String, Value>,
        scheduler: &mut HashMap<String, Value>,
        engine: &mut HashMap<String, Value>,
        serve: &mut HashMap<String, Value>,
        chat: &mut HashMap<String, Value>,
        map: &serde_json::Map<String, Value>,
    ) {
        if let Some(Value::Object(m)) = map.get("model") {
            model.extend(m.clone());
        }
        if let Some(Value::Object(s)) = map.get("scheduler") {
            scheduler.extend(s.clone());
        }
        if let Some(Value::Object(e)) = map.get("engine") {
            engine.extend(e.clone());
        }
        if let Some(Value::Object(s)) = map.get("serve") {
            serve.extend(s.clone());
        }
        if let Some(Value::Object(c)) = map.get("chat") {
            chat.extend(c.clone());
        }
    }

    fn set_nested_key(
        key: &str,
        value: Value,
        model: &mut HashMap<String, Value>,
        scheduler: &mut HashMap<String, Value>,
        engine: &mut HashMap<String, Value>,
        serve: &mut HashMap<String, Value>,
        chat: &mut HashMap<String, Value>,
    ) {
        let parts: Vec<&str> = key.split('.').collect();
        if parts.is_empty() {
            return;
        }

        let target = match parts[0] {
            "model" => model,
            "scheduler" => scheduler,
            "engine" => engine,
            "serve" => serve,
            "chat" => chat,
            _ => return,
        };

        if parts.len() == 1 {
            target.insert(key.to_string(), value);
        } else {
            let nested_key = parts[1..].join(".");
            target.insert(nested_key, value);
        }
    }

    pub fn apply_to_config(&self, config: &mut Config) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(model) = &self.model {
            self.apply_model_args(model, config)?;
        }
        if let Some(scheduler) = &self.scheduler {
            self.apply_scheduler_args(scheduler, config)?;
        }
        if let Some(engine) = &self.engine {
            self.apply_engine_args(engine, config)?;
        }
        if let Some(serve) = &self.serve {
            self.apply_serve_args(serve, config)?;
        }
        if let Some(chat) = &self.chat {
            self.apply_chat_args(chat, config)?;
        }
        Ok(())
    }

    fn apply_model_args(
        &self,
        args: &HashMap<String, Value>,
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (key, value) in args {
            match key.as_str() {
                "model" => config.model.model = self.value_to_string(value)?,
                "tokenizer" => config.model.tokenizer = Some(self.value_to_string(value)?),
                "tokenizer_mode" => config.model.tokenizer_mode = self.value_to_enum(value)?,
                "dtype" => config.model.dtype = self.value_to_enum(value)?,
                "max_model_len" | "max-model-len" => {
                    config.model.max_model_len = Some(self.value_to_usize(value)?)
                }
                "trust_remote_code" | "trust-remote-code" => {
                    config.model.trust_remote_code = self.value_to_bool(value)?
                }
                "quantization" => config.model.quantization = Some(self.value_to_string(value)?),
                "kv_cache_dtype" | "kv-cache-dtype" => {
                    config.model.kv_cache_dtype = Some(self.value_to_string(value)?)
                }
                "served_model_name" | "served-model-name" => {
                    config.model.served_model_name = Some(self.value_to_string(value)?)
                }
                "revision" => config.model.revision = Some(self.value_to_string(value)?),
                "code_revision" | "code-revision" => {
                    config.model.code_revision = Some(self.value_to_string(value)?)
                }
                "tokenizer_revision" | "tokenizer-revision" => {
                    config.model.tokenizer_revision = Some(self.value_to_string(value)?)
                }
                "download_dir" | "download-dir" => {
                    config.model.download_dir = Some(self.value_to_string(value)?)
                }
                "seed" => config.model.seed = self.value_to_usize(value)?,
                "hf_config_path" | "hf-config-path" => {
                    config.model.hf_config_path = Some(self.value_to_string(value)?)
                }
                "max_logprobs" | "max-logprobs" => {
                    config.model.max_logprobs = self.value_to_usize(value)?
                }
                "disable_sliding_window" | "disable-sliding-window" => {
                    config.model.disable_sliding_window = self.value_to_bool(value)?
                }
                "disable_cascade_attn" | "disable-cascade-attn" => {
                    config.model.disable_cascade_attn = self.value_to_bool(value)?
                }
                "min_p" | "min-p" => config.model.min_p = self.value_to_f64(value)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn apply_scheduler_args(
        &self,
        args: &HashMap<String, Value>,
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (key, value) in args {
            match key.as_str() {
                "max_num_seqs" | "max-num-seqs" => {
                    config.scheduler.max_num_seqs = self.value_to_usize(value)?
                }
                "max_num_batched_tokens" | "max-num-batched-tokens" => {
                    config.scheduler.max_num_batched_tokens = self.value_to_usize(value)?
                }
                "enable_continuous_batching" | "enable-continuous-batching" => {
                    config.scheduler.enable_continuous_batching = self.value_to_bool(value)?
                }
                "scheduling_policy" | "scheduling-policy" => {
                    config.scheduler.scheduling_policy = self.value_to_enum(value)?
                }
                "schedule_timeout_ms" | "schedule-timeout-ms" => {
                    config.scheduler.schedule_timeout_ms = self.value_to_usize(value)?
                }
                "dialogue_cache_enabled" | "dialogue-cache-enabled" => {
                    config.scheduler.dialogue_cache_enabled = self.value_to_bool(value)?
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn apply_engine_args(
        &self,
        args: &HashMap<String, Value>,
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (key, value) in args {
            match key.as_str() {
                "runner" => config.engine.runner = self.value_to_enum(value)?,
                "convert" => config.engine.convert = self.value_to_enum(value)?,
                "enforce_eager" | "enforce-eager" => {
                    config.engine.enforce_eager = self.value_to_bool(value)?
                }
                "enable_return_routed_experts" | "enable-return-routed-experts" => {
                    config.engine.enable_return_routed_experts = self.value_to_bool(value)?
                }
                "use_fp64_gumbel" | "use-fp64-gumbel" => {
                    config.engine.use_fp64_gumbel = self.value_to_bool(value)?
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn apply_serve_args(
        &self,
        args: &HashMap<String, Value>,
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if config.serve.is_none() {
            config.serve = Some(ServeConfig::default());
        }
        if let Some(serve) = &mut config.serve {
            for (key, value) in args {
                match key.as_str() {
                    "host" => serve.host = self.value_to_string(value)?,
                    "port" => serve.port = self.value_to_u16(value)?,
                    "log_requests" | "log-requests" => {
                        serve.log_requests = self.value_to_bool(value)?
                    }
                    "api_key" | "api-key" => serve.api_key = Some(self.value_to_string(value)?),
                    "reasoning_parser_enabled" | "reasoning-parser-enabled" => {
                        serve.reasoning_parser_enabled = self.value_to_bool(value)?
                    }
                    "tool_call_parser_enabled" | "tool-call-parser-enabled" => {
                        serve.tool_call_parser_enabled = self.value_to_bool(value)?
                    }
                    "api_server_count" | "api-server-count" => {
                        serve.api_server_count = self.value_to_usize(value)?
                    }
                    "uds" => serve.uds = Some(self.value_to_string(value)?),
                    "ssl_keyfile" | "ssl-keyfile" => {
                        serve.ssl_keyfile = Some(self.value_to_string(value)?)
                    }
                    "ssl_certfile" | "ssl-certfile" => {
                        serve.ssl_certfile = Some(self.value_to_string(value)?)
                    }
                    "ssl_ca_certs" | "ssl-ca-certs" => {
                        serve.ssl_ca_certs = Some(self.value_to_string(value)?)
                    }
                    "allow_credentials" | "allow-credentials" => {
                        serve.allow_credentials = self.value_to_bool(value)?
                    }
                    "allowed_origins" | "allowed-origins" => {
                        serve.allowed_origins = self.value_to_vec_string(value)?
                    }
                    "allowed_methods" | "allowed-methods" => {
                        serve.allowed_methods = self.value_to_vec_string(value)?
                    }
                    "allowed_headers" | "allowed-headers" => {
                        serve.allowed_headers = self.value_to_vec_string(value)?
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn apply_chat_args(
        &self,
        args: &HashMap<String, Value>,
        config: &mut Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if config.chat.is_none() {
            config.chat = Some(ChatConfig::default());
        }
        if let Some(chat) = &mut config.chat {
            for (key, value) in args {
                match key.as_str() {
                    "system_prompt" | "system-prompt" => {
                        chat.system_prompt = Some(self.value_to_string(value)?)
                    }
                    "stream" => chat.stream = self.value_to_bool(value)?,
                    "max_turns" | "max-turns" => chat.max_turns = Some(self.value_to_usize(value)?),
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn value_to_string(&self, value: &Value) -> Result<String, Box<dyn std::error::Error>> {
        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err("Expected string value".into()),
        }
    }

    fn value_to_usize(&self, value: &Value) -> Result<usize, Box<dyn std::error::Error>> {
        match value {
            Value::Number(n) => n
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| "Expected usize value".into()),
            _ => Err("Expected usize value".into()),
        }
    }

    fn value_to_u16(&self, value: &Value) -> Result<u16, Box<dyn std::error::Error>> {
        match value {
            Value::Number(n) => n
                .as_u64()
                .map(|v| v as u16)
                .ok_or_else(|| "Expected u16 value".into()),
            _ => Err("Expected u16 value".into()),
        }
    }

    fn value_to_f64(&self, value: &Value) -> Result<f64, Box<dyn std::error::Error>> {
        match value {
            Value::Number(n) => n.as_f64().ok_or_else(|| "Expected f64 value".into()),
            _ => Err("Expected f64 value".into()),
        }
    }

    fn value_to_bool(&self, value: &Value) -> Result<bool, Box<dyn std::error::Error>> {
        match value {
            Value::Bool(b) => Ok(*b),
            _ => Err("Expected bool value".into()),
        }
    }

    fn value_to_vec_string(
        &self,
        value: &Value,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        match value {
            Value::Array(arr) => arr.iter().map(|v| self.value_to_string(v)).collect(),
            Value::String(s) => Ok(s.split(',').map(|s| s.trim().to_string()).collect()),
            _ => Err("Expected array or comma-separated string".into()),
        }
    }

    fn value_to_enum<T: ValueEnum>(&self, value: &Value) -> Result<T, Box<dyn std::error::Error>> {
        let s = self.value_to_string(value)?;
        T::from_str(&s, false).map_err(|_| format!("Invalid enum value: {}", s).into())
    }
}

impl Config {
    fn apply_shared_args(&mut self, shared: &SharedArgs) {
        if let Some(min_p) = shared.min_p {
            self.model.min_p = min_p;
        }
        if let Some(model) = &shared.model {
            self.model.model = model.clone();
        }
        if let Some(tokenizer) = &shared.tokenizer {
            self.model.tokenizer = Some(tokenizer.clone());
        }
        if let Some(tokenizer_mode) = shared.tokenizer_mode {
            self.model.tokenizer_mode = tokenizer_mode;
        }
        if let Some(dtype) = shared.dtype {
            self.model.dtype = dtype;
        }
        if let Some(max_model_len) = shared.max_model_len {
            self.model.max_model_len = Some(max_model_len);
        }
        if let Some(trust_remote_code) = shared.trust_remote_code {
            self.model.trust_remote_code = trust_remote_code;
        }
        if let Some(quantization) = &shared.quantization {
            self.model.quantization = Some(quantization.clone());
        }
        if let Some(kv_cache_dtype) = &shared.kv_cache_dtype {
            self.model.kv_cache_dtype = Some(kv_cache_dtype.clone());
        }
        if let Some(served_model_name) = &shared.served_model_name {
            self.model.served_model_name = Some(served_model_name.clone());
        }
        if let Some(revision) = &shared.revision {
            self.model.revision = Some(revision.clone());
        }
        if let Some(code_revision) = &shared.code_revision {
            self.model.code_revision = Some(code_revision.clone());
        }
        if let Some(tokenizer_revision) = &shared.tokenizer_revision {
            self.model.tokenizer_revision = Some(tokenizer_revision.clone());
        }
        if let Some(download_dir) = &shared.download_dir {
            self.model.download_dir = Some(download_dir.clone());
        }
        if let Some(seed) = shared.seed {
            self.model.seed = seed;
        }
        if let Some(hf_config_path) = &shared.hf_config_path {
            self.model.hf_config_path = Some(hf_config_path.clone());
        }
        if let Some(max_logprobs) = shared.max_logprobs {
            self.model.max_logprobs = max_logprobs;
        }
        if let Some(disable_sliding_window) = shared.disable_sliding_window {
            self.model.disable_sliding_window = disable_sliding_window;
        }
        if let Some(disable_cascade_attn) = shared.disable_cascade_attn {
            self.model.disable_cascade_attn = disable_cascade_attn;
        }
        if let Some(runner) = shared.runner {
            self.engine.runner = runner;
        }
        if let Some(convert) = shared.convert {
            self.engine.convert = convert;
        }
        if let Some(enforce_eager) = shared.enforce_eager {
            self.engine.enforce_eager = enforce_eager;
        }
        if let Some(enable_return_routed_experts) = shared.enable_return_routed_experts {
            self.engine.enable_return_routed_experts = enable_return_routed_experts;
        }
        if let Some(use_fp64_gumbel) = shared.use_fp64_gumbel {
            self.engine.use_fp64_gumbel = use_fp64_gumbel;
        }
        if let Some(max_num_seqs) = shared.max_num_seqs {
            self.scheduler.max_num_seqs = max_num_seqs;
        }
        if let Some(max_num_batched_tokens) = shared.max_num_batched_tokens {
            self.scheduler.max_num_batched_tokens = max_num_batched_tokens;
        }
        if let Some(enable_continuous_batching) = shared.enable_continuous_batching {
            self.scheduler.enable_continuous_batching = enable_continuous_batching;
        }
        if let Some(scheduling_policy) = shared.scheduling_policy {
            self.scheduler.scheduling_policy = scheduling_policy;
        }
        if let Some(schedule_timeout_ms) = shared.schedule_timeout_ms {
            self.scheduler.schedule_timeout_ms = schedule_timeout_ms;
        }
        if let Some(dialogue_cache_enabled) = shared.dialogue_cache_enabled {
            self.scheduler.dialogue_cache_enabled = dialogue_cache_enabled;
        }
    }

    fn apply_serve_args(&mut self, args: ServeArgs) {
        self.apply_shared_args(&args.shared);
        self.chat = None;
        self.serve.get_or_insert_with(ServeConfig::default);
        if let Some(serve) = self.serve.as_mut() {
            if let Some(host) = args.host {
                serve.host = host;
            }
            if let Some(port) = args.port {
                serve.port = port;
            }
            if let Some(log_requests) = args.log_requests {
                serve.log_requests = log_requests;
            }
            if let Some(api_key) = args.api_key {
                serve.api_key = Some(api_key);
            }
            if let Some(reasoning_parser_enabled) = args.reasoning_parser_enabled {
                serve.reasoning_parser_enabled = reasoning_parser_enabled;
            }
            if let Some(tool_call_parser_enabled) = args.tool_call_parser_enabled {
                serve.tool_call_parser_enabled = tool_call_parser_enabled;
            }
            if let Some(api_server_count) = args.api_server_count {
                serve.api_server_count = api_server_count;
            }
            if let Some(uds) = args.uds {
                serve.uds = Some(uds);
            }
            if let Some(ssl_keyfile) = args.ssl_keyfile {
                serve.ssl_keyfile = Some(ssl_keyfile);
            }
            if let Some(ssl_certfile) = args.ssl_certfile {
                serve.ssl_certfile = Some(ssl_certfile);
            }
            if let Some(ssl_ca_certs) = args.ssl_ca_certs {
                serve.ssl_ca_certs = Some(ssl_ca_certs);
            }
            if let Some(allow_credentials) = args.allow_credentials {
                serve.allow_credentials = allow_credentials;
            }
            if let Some(allowed_origins) = args.allowed_origins {
                serve.allowed_origins = allowed_origins;
            }
            if let Some(allowed_methods) = args.allowed_methods {
                serve.allowed_methods = allowed_methods;
            }
            if let Some(allowed_headers) = args.allowed_headers {
                serve.allowed_headers = allowed_headers;
            }
        }
    }

    fn apply_chat_args(&mut self, args: ChatArgs) {
        self.apply_shared_args(&args.shared);
        self.serve = None;
        self.chat.get_or_insert_with(ChatConfig::default);
        if let Some(chat) = self.chat.as_mut() {
            if let Some(system_prompt) = args.system_prompt {
                chat.system_prompt = Some(system_prompt);
            }
            if let Some(stream) = args.stream {
                chat.stream = stream;
            }
            if let Some(max_turns) = args.max_turns {
                chat.max_turns = Some(max_turns);
            }
        }
    }

    pub fn from_serve_args(
        args: ServeArgs,
        json_args: Option<JsonArgs>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = if let Some(path) = args.config.as_ref() {
            Self::load_from_file(path)?
        } else {
            Self::empty(Command::Serve)
        };

        config.command = Command::Serve;
        config.apply_serve_args(args);

        if let Some(json_args) = json_args {
            json_args.apply_to_config(&mut config)?;
        }

        config.validate()?;
        Ok(config)
    }

    pub fn from_chat_args(
        args: ChatArgs,
        json_args: Option<JsonArgs>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = if let Some(path) = args.config.as_ref() {
            Self::load_from_file(path)?
        } else {
            Self::empty(Command::Chat)
        };

        config.command = Command::Chat;
        config.apply_chat_args(args);

        if let Some(json_args) = json_args {
            json_args.apply_to_config(&mut config)?;
        }

        config.validate()?;
        Ok(config)
    }

    pub fn from_yaml_and_cli<P: AsRef<std::path::Path>>(
        filename: P,
        cli: Cli,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = Self::load_from_file(filename)?;

        let json_args = if !cli.json_arg.is_empty() {
            Some(JsonArgs::parse(&cli.json_arg)?)
        } else {
            None
        };

        match cli.command {
            CliCommand::Serve(args) => {
                config.command = Command::Serve;
                config.apply_serve_args(args);
            }
            CliCommand::Chat(args) => {
                config.command = Command::Chat;
                config.apply_chat_args(args);
            }
        }

        if let Some(json_args) = json_args {
            json_args.apply_to_config(&mut config)?;
        }

        config.validate()?;
        Ok(config)
    }

    pub fn from_cli(cli: Cli) -> Result<Self, Box<dyn std::error::Error>> {
        let json_args = if !cli.json_arg.is_empty() {
            Some(JsonArgs::parse(&cli.json_arg)?)
        } else {
            None
        };

        let command = cli.command;
        let config_path = cli.config;

        let mut config = match &command {
            CliCommand::Serve(args) => Self::from_serve_args(args.clone(), None)?,
            CliCommand::Chat(args) => Self::from_chat_args(args.clone(), None)?,
        };

        if let Some(path) = config_path {
            let vllm_config = VllmConfigFile::load_from_file(&path)?;
            vllm_config.apply_to_config(&mut config)?;
        }

        match command {
            CliCommand::Serve(args) => {
                config.apply_serve_args(args);
                config.command = Command::Serve;
            }
            CliCommand::Chat(args) => {
                config.apply_chat_args(args);
                config.command = Command::Chat;
            }
        }

        if let Some(json_args) = json_args {
            json_args.apply_to_config(&mut config)?;
        }

        config.validate()?;
        Ok(config)
    }
}
