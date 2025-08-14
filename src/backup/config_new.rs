/*!
 * Model Configuration
 *
 * This module provides the Config struct that corresponds exactly to the
 * Python ModelArgs class, ensuring compatibility between Rust and Python APIs.
 */

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

use crate::kernel::generic::from_f32::FromF32;

/// Model configuration struct that corresponds to Python ModelArgs
///
/// This struct contains all the parameters needed to configure a Llama model.
/// It provides serialization/deserialization to/from JSON files and ensures
/// compatibility with the Python API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    // Core model architecture
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,

    // Sequence and batch configuration
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub max_batch_size: usize,

    // Architecture details
    pub attention_bias: Option<bool>,
    pub attention_dropout: Option<f32>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub pad_token_id: Option<usize>,
    pub hidden_act: String,
    pub initializer_range: f32,
    pub model_type: String,
    pub pretraining_tp: usize,
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,

    // Computed fields
    pub attention_head_size: usize,

    // Legacy fields for compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub _name_or_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architectures: Option<Vec<String>>,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Create a new Config with default values
    ///
    /// These defaults correspond to a typical Llama-7B model configuration.
    pub fn new() -> Self {
        let hidden_size = 4096;
        let num_attention_heads = 32;

        Config {
            // Core architecture
            hidden_size,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads,
            num_key_value_heads: None, // Will default to num_attention_heads
            intermediate_size: None,   // Will default to 4 * hidden_size
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,

            // Sequence and batch
            batch_size: 1,
            max_seq_len: 2048,
            max_batch_size: 32,

            // Architecture details
            attention_bias: Some(false),
            attention_dropout: Some(0.0),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: None,
            hidden_act: "silu".to_string(),
            initializer_range: 0.02,
            model_type: "llama".to_string(),
            pretraining_tp: 1,
            rope_scaling: None,
            tie_word_embeddings: false,
            torch_dtype: "float16".to_string(),
            transformers_version: "4.21.0.dev0".to_string(),
            use_cache: true,

            // Computed
            attention_head_size: hidden_size / num_attention_heads,

            // Legacy
            _name_or_path: None,
            architectures: None,
        }
    }

    /// Load configuration from a JSON file
    ///
    /// # Arguments
    /// * `filename` - Path to the config.json file
    ///
    /// # Returns
    /// Result containing the loaded Config or an error
    pub fn load_model_config<P: AsRef<Path>>(
        &mut self,
        filename: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        // Parse JSON into a temporary config
        let tmp_config: serde_json::Value = serde_json::from_reader(reader)?;

        // Update fields from the loaded config
        if let Some(hidden_size) = tmp_config.get("hidden_size").and_then(|v| v.as_u64()) {
            self.hidden_size = hidden_size as usize;
        }

        if let Some(vocab_size) = tmp_config.get("vocab_size").and_then(|v| v.as_u64()) {
            self.vocab_size = vocab_size as usize;
        }

        if let Some(num_hidden_layers) =
            tmp_config.get("num_hidden_layers").and_then(|v| v.as_u64())
        {
            self.num_hidden_layers = num_hidden_layers as usize;
        }

        if let Some(num_attention_heads) = tmp_config
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
        {
            self.num_attention_heads = num_attention_heads as usize;
        }

        if let Some(num_key_value_heads) = tmp_config
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
        {
            self.num_key_value_heads = Some(num_key_value_heads as usize);
        }

        if let Some(intermediate_size) =
            tmp_config.get("intermediate_size").and_then(|v| v.as_u64())
        {
            self.intermediate_size = Some(intermediate_size as usize);
        }

        if let Some(max_position_embeddings) = tmp_config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
        {
            self.max_position_embeddings = max_position_embeddings as usize;
        }

        if let Some(rms_norm_eps) = tmp_config.get("rms_norm_eps").and_then(|v| v.as_f64()) {
            self.rms_norm_eps = rms_norm_eps as f32;
        }

        if let Some(bos_token_id) = tmp_config.get("bos_token_id").and_then(|v| v.as_u64()) {
            self.bos_token_id = Some(bos_token_id as usize);
        }

        if let Some(eos_token_id) = tmp_config.get("eos_token_id").and_then(|v| v.as_u64()) {
            self.eos_token_id = Some(eos_token_id as usize);
        }

        if let Some(pad_token_id) = tmp_config.get("pad_token_id").and_then(|v| v.as_u64()) {
            self.pad_token_id = Some(pad_token_id as usize);
        }

        if let Some(hidden_act) = tmp_config.get("hidden_act").and_then(|v| v.as_str()) {
            self.hidden_act = hidden_act.to_string();
        }

        if let Some(initializer_range) =
            tmp_config.get("initializer_range").and_then(|v| v.as_f64())
        {
            self.initializer_range = initializer_range as f32;
        }

        if let Some(model_type) = tmp_config.get("model_type").and_then(|v| v.as_str()) {
            self.model_type = model_type.to_string();
        }

        if let Some(pretraining_tp) = tmp_config.get("pretraining_tp").and_then(|v| v.as_u64()) {
            self.pretraining_tp = pretraining_tp as usize;
        }

        if let Some(tie_word_embeddings) = tmp_config
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
        {
            self.tie_word_embeddings = tie_word_embeddings;
        }

        if let Some(torch_dtype) = tmp_config.get("torch_dtype").and_then(|v| v.as_str()) {
            self.torch_dtype = torch_dtype.to_string();
        }

        if let Some(transformers_version) = tmp_config
            .get("transformers_version")
            .and_then(|v| v.as_str())
        {
            self.transformers_version = transformers_version.to_string();
        }

        if let Some(use_cache) = tmp_config.get("use_cache").and_then(|v| v.as_bool()) {
            self.use_cache = use_cache;
        }

        // Handle rope_scaling as a generic JSON value
        if let Some(rope_scaling) = tmp_config.get("rope_scaling") {
            if !rope_scaling.is_null() {
                let mut rope_map = HashMap::new();
                if let Some(obj) = rope_scaling.as_object() {
                    for (k, v) in obj {
                        rope_map.insert(k.clone(), v.clone());
                    }
                }
                self.rope_scaling = Some(rope_map);
            }
        }

        // Set defaults for optional fields
        if self.num_key_value_heads.is_none() {
            self.num_key_value_heads = Some(self.num_attention_heads);
        }

        if self.intermediate_size.is_none() {
            self.intermediate_size = Some(4 * self.hidden_size);
        }

        // Recompute derived fields
        self.attention_head_size = self.hidden_size / self.num_attention_heads;

        Ok(())
    }

    /// Load compile configuration from a JSON file
    ///
    /// This loads additional compilation/runtime specific parameters.
    ///
    /// # Arguments
    /// * `filename` - Path to the compile config JSON file
    pub fn load_compile_config<P: AsRef<Path>>(
        &mut self,
        filename: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        let compile_config: serde_json::Value = serde_json::from_reader(reader)?;

        // Load compile-specific parameters
        if let Some(batch_size) = compile_config.get("batch_size").and_then(|v| v.as_u64()) {
            self.batch_size = batch_size as usize;
        }

        if let Some(max_seq_len) = compile_config.get("max_seq_len").and_then(|v| v.as_u64()) {
            self.max_seq_len = max_seq_len as usize;
        }

        if let Some(max_batch_size) = compile_config
            .get("max_batch_size")
            .and_then(|v| v.as_u64())
        {
            self.max_batch_size = max_batch_size as usize;
        }

        Ok(())
    }

    /// Save configuration to a JSON file
    ///
    /// # Arguments
    /// * `filename` - Path where to save the config
    pub fn save_config<P: AsRef<Path>>(
        &self,
        filename: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(filename)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Convert to a HashMap for easier manipulation
    pub fn to_map(&self) -> HashMap<String, serde_json::Value> {
        let json = serde_json::to_value(self).unwrap();
        json.as_object().unwrap().clone()
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err("hidden_size must be greater than 0".to_string());
        }

        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be greater than 0".to_string());
        }

        if self.hidden_size % self.num_attention_heads != 0 {
            return Err("hidden_size must be divisible by num_attention_heads".to_string());
        }

        if self.num_hidden_layers == 0 {
            return Err("num_hidden_layers must be greater than 0".to_string());
        }

        if self.vocab_size == 0 {
            return Err("vocab_size must be greater than 0".to_string());
        }

        if self.max_position_embeddings == 0 {
            return Err("max_position_embeddings must be greater than 0".to_string());
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if self.num_attention_heads % num_kv_heads != 0 {
                return Err(
                    "num_attention_heads must be divisible by num_key_value_heads".to_string(),
                );
            }
        }

        Ok(())
    }

    /// Get the effective number of key-value heads
    pub fn get_num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the effective intermediate size
    pub fn get_intermediate_size(&self) -> usize {
        self.intermediate_size.unwrap_or(4 * self.hidden_size)
    }

    /// Calculate approximate memory usage in bytes
    pub fn estimate_memory_usage(&self, precision_bytes: usize) -> usize {
        let embed_params = self.vocab_size * self.hidden_size;
        let attention_params = 4 * self.hidden_size * self.hidden_size;
        let mlp_params = 3 * self.hidden_size * self.get_intermediate_size();
        let norm_params = 2 * self.hidden_size;
        let layer_params = attention_params + mlp_params + norm_params;
        let total_layer_params = self.num_hidden_layers * layer_params;
        let other_params = self.hidden_size + self.vocab_size * self.hidden_size;

        let total_params = embed_params + total_layer_params + other_params;
        total_params * precision_bytes
    }
}

/// Helper functions for config manipulation
pub mod config_utils {
    use super::*;

    /// Create a config for a specific model size
    pub fn create_config_for_size(model_size: &str) -> Config {
        let mut config = Config::new();

        match model_size.to_lowercase().as_str() {
            "7b" => {
                config.hidden_size = 4096;
                config.num_hidden_layers = 32;
                config.num_attention_heads = 32;
                config.intermediate_size = Some(11008);
            }
            "13b" => {
                config.hidden_size = 5120;
                config.num_hidden_layers = 40;
                config.num_attention_heads = 40;
                config.intermediate_size = Some(13824);
            }
            "30b" => {
                config.hidden_size = 6656;
                config.num_hidden_layers = 60;
                config.num_attention_heads = 52;
                config.intermediate_size = Some(17920);
            }
            "65b" => {
                config.hidden_size = 8192;
                config.num_hidden_layers = 80;
                config.num_attention_heads = 64;
                config.intermediate_size = Some(22016);
            }
            _ => {
                // Default to 7B configuration
            }
        }

        // Recompute derived fields
        config.attention_head_size = config.hidden_size / config.num_attention_heads;

        config
    }

    /// Compare two configurations
    pub fn configs_equal(config1: &Config, config2: &Config) -> bool {
        config1.hidden_size == config2.hidden_size
            && config1.vocab_size == config2.vocab_size
            && config1.num_hidden_layers == config2.num_hidden_layers
            && config1.num_attention_heads == config2.num_attention_heads
            && config1.get_num_key_value_heads() == config2.get_num_key_value_heads()
            && config1.get_intermediate_size() == config2.get_intermediate_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::new();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.attention_head_size, 128);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::new();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();

        assert_eq!(config.hidden_size, deserialized.hidden_size);
        assert_eq!(config.vocab_size, deserialized.vocab_size);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::new();
        assert!(config.validate().is_ok());

        config.hidden_size = 0;
        assert!(config.validate().is_err());

        config.hidden_size = 4096;
        config.num_attention_heads = 0;
        assert!(config.validate().is_err());

        config.num_attention_heads = 33; // Not divisible
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_size_configs() {
        let config_7b = config_utils::create_config_for_size("7b");
        assert_eq!(config_7b.hidden_size, 4096);
        assert_eq!(config_7b.num_hidden_layers, 32);

        let config_13b = config_utils::create_config_for_size("13b");
        assert_eq!(config_13b.hidden_size, 5120);
        assert_eq!(config_13b.num_hidden_layers, 40);
    }

    #[test]
    fn test_memory_estimation() {
        let config = Config::new();
        let memory_f16 = config.estimate_memory_usage(2);
        let memory_f32 = config.estimate_memory_usage(4);

        assert!(memory_f16 > 0);
        assert_eq!(memory_f32, memory_f16 * 2);
    }
}
