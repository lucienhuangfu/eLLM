use super::types::{
    ChatConfig, Command, Config, ModelConfig, ResolvedConfig, ResolvedModelConfig, ServeConfig,
};
use std::{fs::File, io::BufReader, path::Path};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("model.model cannot be empty")]
    EmptyModel,
    #[error("model.max_model_len must be greater than 0 when provided")]
    InvalidMaxModelLen,
    #[error("model.served_model_name cannot be empty when provided")]
    InvalidServedModelName,
    #[error("model.{0} cannot be empty when provided")]
    InvalidRevision(&'static str),
    #[error("model.download_dir cannot be empty when provided")]
    InvalidDownloadDir,
    #[error("scheduler.max_num_seqs must be greater than 0")]
    InvalidMaxNumSeqs,
    #[error("scheduler.max_num_batched_tokens must be greater than 0")]
    InvalidMaxNumBatchedTokens,
    #[error("scheduler.prefill_chunk_size must be greater than 0 when provided")]
    InvalidPrefillChunkSize,
    #[error("scheduler.prefill_chunk_size ({prefill_chunk_size}) cannot exceed max_num_batched_tokens ({max_num_batched_tokens})")]
    PrefillChunkTooLarge {
        prefill_chunk_size: usize,
        max_num_batched_tokens: usize,
    },
    #[error("scheduler.max_num_batched_tokens must be greater than or equal to max_num_seqs")]
    InconsistentBatchLimits {
        max_num_seqs: usize,
        max_num_batched_tokens: usize,
    },
    #[error("serve.host cannot be empty")]
    InvalidHost,
    #[error("serve.port must be greater than 0")]
    InvalidPort,
    #[error("command `{0}` requires a matching `{1}` section")]
    MissingCommandSection(&'static str, &'static str),
}

impl Config {
    pub fn empty(command: Command) -> Self {
        Self {
            command,
            model: ModelConfig::default(),
            scheduler: super::types::SchedulerConfig::default(),
            serve: Some(ServeConfig::default()),
            chat: Some(ChatConfig::default()),
        }
    }

    fn validate_common(&self) -> Result<(), ConfigError> {
        if self.model.model.trim().is_empty() {
            return Err(ConfigError::EmptyModel);
        }

        if let Some(max_model_len) = self.model.max_model_len {
            if max_model_len == 0 {
                return Err(ConfigError::InvalidMaxModelLen);
            }
        }

        if let Some(served_model_name) = self.model.served_model_name.as_ref() {
            if served_model_name.trim().is_empty() {
                return Err(ConfigError::InvalidServedModelName);
            }
        }

        if let Some(revision) = self.model.revision.as_ref() {
            if revision.trim().is_empty() {
                return Err(ConfigError::InvalidRevision("revision"));
            }
        }

        if let Some(tokenizer_revision) = self.model.tokenizer_revision.as_ref() {
            if tokenizer_revision.trim().is_empty() {
                return Err(ConfigError::InvalidRevision("tokenizer_revision"));
            }
        }

        if let Some(download_dir) = self.model.download_dir.as_ref() {
            if download_dir.trim().is_empty() {
                return Err(ConfigError::InvalidDownloadDir);
            }
        }

        if self.scheduler.max_num_seqs == 0 {
            return Err(ConfigError::InvalidMaxNumSeqs);
        }

        if self.scheduler.max_num_batched_tokens == 0 {
            return Err(ConfigError::InvalidMaxNumBatchedTokens);
        }

        if matches!(self.scheduler.prefill_chunk_size, Some(0)) {
            return Err(ConfigError::InvalidPrefillChunkSize);
        }

        if let Some(prefill_chunk_size) = self.scheduler.prefill_chunk_size {
            if prefill_chunk_size > self.scheduler.max_num_batched_tokens {
                return Err(ConfigError::PrefillChunkTooLarge {
                    prefill_chunk_size,
                    max_num_batched_tokens: self.scheduler.max_num_batched_tokens,
                });
            }
        }

        if self.scheduler.max_num_batched_tokens < self.scheduler.max_num_seqs {
            return Err(ConfigError::InconsistentBatchLimits {
                max_num_seqs: self.scheduler.max_num_seqs,
                max_num_batched_tokens: self.scheduler.max_num_batched_tokens,
            });
        }

        Ok(())
    }

    fn validate_mode_section(&self) -> Result<(), ConfigError> {
        match self.command {
            Command::Serve => {
                if self.serve.is_none() {
                    return Err(ConfigError::MissingCommandSection(
                        "serve",
                        "serve config",
                    ));
                }
                if let Some(serve) = self.serve.as_ref() {
                    if serve.host.trim().is_empty() {
                        return Err(ConfigError::InvalidHost);
                    }
                    if serve.port == 0 {
                        return Err(ConfigError::InvalidPort);
                    }
                }
            }
            Command::Chat => {
                if self.chat.is_none() {
                    return Err(ConfigError::MissingCommandSection(
                        "chat",
                        "chat config",
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validate_common()?;
        self.validate_mode_section()
    }

    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let config: Config = serde_yaml::from_reader(reader)?;
        config.validate_common()?;
        Ok(config)
    }

    pub fn resolve(&self) -> Result<ResolvedConfig, ConfigError> {
        self.validate()?;
        Ok(ResolvedConfig {
            command: self.command,
            model: ResolvedModelConfig {
                raw: self.model.clone(),
                served_model_name: self
                    .model
                    .served_model_name
                    .clone()
                    .unwrap_or_else(|| super::types::infer_served_model_name(&self.model.model)),
                effective_tokenizer: self
                    .model
                    .tokenizer
                    .clone()
                    .unwrap_or_else(|| self.model.model.clone()),
            },
            scheduler: self.scheduler.clone(),
            serve: self.serve.clone(),
            chat: self.chat.clone(),
        })
    }

    pub fn effective_mode(&self) -> &'static str {
        match self.command {
            Command::Serve => "serve",
            Command::Chat => "chat",
        }
    }
}

impl ResolvedConfig {
    pub fn served_model_name(&self) -> &str {
        &self.model.served_model_name
    }

    pub fn effective_tokenizer(&self) -> &str {
        &self.model.effective_tokenizer
    }
}