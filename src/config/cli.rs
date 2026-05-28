use super::types::{ChatConfig, Command, Config, ModelDtype, SchedulingPolicy, ServeConfig};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Clone, Parser)]
#[command(name = "ellm")]
#[command(about = "Simplified LLM runtime configuration CLI")]
pub struct Cli {
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
    #[arg(long = "min-p", default_value = "0.0")]
    pub min_p: Option<f64>,
    #[arg(long = "model")]
    pub model: Option<String>,
    #[arg(long = "tokenizer")]
    pub tokenizer: Option<String>,
    #[arg(long = "dtype", value_enum)]
    pub dtype: Option<ModelDtype>,
    #[arg(long = "max-model-len")]
    pub max_model_len: Option<usize>,
    #[arg(long = "trust-remote-code", num_args = 0..=1, default_missing_value = "true")]
    pub trust_remote_code: Option<bool>,
    #[arg(long = "quantization")]
    pub quantization: Option<String>,
    #[arg(long = "kv-cache-dtype")]
    pub kv_cache_dtype: Option<String>,
    #[arg(long = "served-model-name")]
    pub served_model_name: Option<String>,
    #[arg(long = "revision")]
    pub revision: Option<String>,
    #[arg(long = "tokenizer-revision")]
    pub tokenizer_revision: Option<String>,
    #[arg(long = "download-dir")]
    pub download_dir: Option<String>,
    #[arg(long = "max-num-seqs")]
    pub max_num_seqs: Option<usize>,
    #[arg(long = "max-num-batched-tokens")]
    pub max_num_batched_tokens: Option<usize>,
    #[arg(long = "enable-continuous-batching", num_args = 0..=1, default_missing_value = "true")]
    pub enable_continuous_batching: Option<bool>,
    #[arg(long = "prefill-chunk-size")]
    pub prefill_chunk_size: Option<usize>,
    #[arg(long = "scheduling-policy", value_enum)]
    pub scheduling_policy: Option<SchedulingPolicy>,
}

#[derive(Debug, Clone, Args)]
pub struct ServeArgs {
    #[command(flatten)]
    pub shared: SharedArgs,
    #[arg(long = "config")]
    pub config: Option<PathBuf>,
    #[arg(long = "host")]
    pub host: Option<String>,
    #[arg(long = "port")]
    pub port: Option<u16>,
    #[arg(long = "log-requests", num_args = 0..=1, default_missing_value = "true")]
    pub log_requests: Option<bool>,
    #[arg(long = "api-key")]
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub struct ChatArgs {
    #[command(flatten)]
    pub shared: SharedArgs,
    #[arg(long = "config")]
    pub config: Option<PathBuf>,
    #[arg(long = "system-prompt")]
    pub system_prompt: Option<String>,
    #[arg(long = "stream", num_args = 0..=1, default_missing_value = "true")]
    pub stream: Option<bool>,
    #[arg(long = "max-turns")]
    pub max_turns: Option<usize>,
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
        if let Some(tokenizer_revision) = &shared.tokenizer_revision {
            self.model.tokenizer_revision = Some(tokenizer_revision.clone());
        }
        if let Some(download_dir) = &shared.download_dir {
            self.model.download_dir = Some(download_dir.clone());
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
        if let Some(prefill_chunk_size) = shared.prefill_chunk_size {
            self.scheduler.prefill_chunk_size = Some(prefill_chunk_size);
        }
        if let Some(scheduling_policy) = shared.scheduling_policy {
            self.scheduler.scheduling_policy = scheduling_policy;
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

    pub fn from_serve_args(args: ServeArgs) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = if let Some(path) = args.config.as_ref() {
            Self::load_from_file(path)?
        } else {
            Self::empty(Command::Serve)
        };

        config.command = Command::Serve;
        config.apply_serve_args(args);
        config.validate()?;
        Ok(config)
    }

    pub fn from_chat_args(args: ChatArgs) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = if let Some(path) = args.config.as_ref() {
            Self::load_from_file(path)?
        } else {
            Self::empty(Command::Chat)
        };

        config.command = Command::Chat;
        config.apply_chat_args(args);
        config.validate()?;
        Ok(config)
    }

    pub fn from_yaml_and_cli<P: AsRef<std::path::Path>>(
        filename: P,
        cli: Cli,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = Self::load_from_file(filename)?;
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

        config.validate()?;
        Ok(config)
    }

    pub fn from_cli(cli: Cli) -> Result<Self, Box<dyn std::error::Error>> {
        match cli.command {
            CliCommand::Serve(args) => Self::from_serve_args(args),
            CliCommand::Chat(args) => Self::from_chat_args(args),
        }
    }
}
