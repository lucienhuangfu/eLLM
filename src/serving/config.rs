use std::env;

pub struct ServingConfig {
    pub model_dir: String,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub chunk_size: usize,
    pub schedule_timeout_ms: usize,
}

impl ServingConfig {
    pub fn new(model_dir: String) -> Self {
        Self {
            model_dir,
            batch_size: parse_env_usize("ELLM_BATCH_SIZE", 3),
            sequence_length: parse_env_usize("ELLM_SEQUENCE_LENGTH", 128),
            chunk_size: parse_env_usize("ELLM_CHUNK_SIZE", 64),
            schedule_timeout_ms: parse_env_usize("ELLM_SCHEDULE_TIMEOUT_MS", 10),
        }
    }
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}
