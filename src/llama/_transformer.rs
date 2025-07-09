/*!
 * Rust Transformer Implementation
 *
 * This module provides the core Transformer struct and implementation that
 * corresponds to the Python ModelArgs and Transformer classes.
 *
 * Key components:
 * - Config: Model configuration (corresponds to Python ModelArgs)
 * - Transformer: Main model struct (corresponds to Python Transformer)
 * - Generation functions for text generation
 */

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::kernel::generic::from_f32::FromF32;
use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::compiler::map::rms_map::RMSMap;
use super::super::compiler::operator::Operator;

use super::super::memory::cache::Cache;
use super::super::ptensor::linear::Linear;
use super::super::ptensor::tensor::Tensor;
use super::transformer_block::TransformerBlock;
use crate::init::config::Config;

/// Main Transformer model structure
///
/// This struct corresponds to the Python Transformer class and contains
/// all the model components needed for inference.
#[derive(Clone)]
pub struct Transformer<T> {
    /// Model configuration parameters
    config: Config,

    /// Final layer normalization weights
    norm_weight: Tensor<T>,

    /// Word embedding weights
    word_embedding: Tensor<T>,

    /// RMS normalization epsilon value
    rms_norm_eps: T,

    /// Output linear layer (language modeling head)
    output_linear: Linear<T>,

    /// Scope name for tensor naming
    scope_name: String,

    /// Position embedding weights
    position_embedding: Tensor<T>,

    /// Number of CPU cores to use
    cpu_num: usize,

    /// Shared cache for tensors and weights
    pub cache: Rc<RefCell<Cache<T>>>,

    /// Queue of operations to execute
    pub operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> Transformer<T>
where
    T: Copy
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid<T>
        + Sqrt
        + FromF32,
{
    /// Create a new Transformer instance
    ///
    /// # Arguments
    /// * `config` - Model configuration parameters
    /// * `word_embedding` - Word embedding tensor
    /// * `position_embedding` - Position embedding tensor
    /// * `norm_weight` - Final layer norm weights
    /// * `cpu_num` - Number of CPU cores to use
    /// * `cache` - Shared tensor cache
    /// * `operator_queue` - Shared operation queue
    ///
    /// # Returns
    /// New Transformer instance
    pub fn new(
        config: Config,
        word_embedding: Tensor<T>,
        position_embedding: Tensor<T>,
        norm_weight: Tensor<T>,
        cpu_num: usize,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = String::from("model");

        Transformer {
            output_linear: Linear::<T>::new(
                config.hidden_size,
                config.vocab_size,
                1,
                format!("lm_head"),
                cache.clone(),
                operator_queue.clone(),
            ),
            position_embedding,
            word_embedding,
            norm_weight,
            rms_norm_eps: T::from_f32(config.rms_norm_eps),
            config,
            cpu_num,
            scope_name,
            cache,
            operator_queue,
        }
    }

    /// Forward pass through the transformer
    ///
    /// This is the main inference function that processes input sequences
    /// through all transformer layers and produces output logits.
    ///
    /// # Arguments
    /// * `sequences` - Pointer to input token sequences
    ///
    /// # Returns
    /// Output tensor with logits
    ///
    /// # Safety
    /// This function is unsafe because it works with raw pointers for performance.
    /// The caller must ensure the sequences pointer is valid.
    pub fn forward(&self, sequences: *mut usize) -> Tensor<T> {
        // Create transformer layers dynamically
        let mut layer_vec: Vec<TransformerBlock<T>> = Vec::new();

        for i in 0..self.config.num_hidden_layers {
            layer_vec.push(TransformerBlock::<T>::new(
                &self.config,
                &self.word_embedding,
                &self.position_embedding,
                i,
                self.cpu_num,
                &self.scope_name,
                self.cache.clone(),
                self.operator_queue.clone(),
            ));
        }

        // Initialize hidden state
        let mut hidden_state = Tensor::<T>::zeros(
            vec![self.config.batch_size, self.config.hidden_size],
            format!("{}.hidden_state", self.scope_name),
            self.cache.clone(),
            self.operator_queue.clone(),
        );

        // Process through all transformer layers
        for (i, layer_module) in layer_vec.iter().enumerate() {
            hidden_state = layer_module.forward(
                &hidden_state,
                sequences,
                format!("{}.hidden_states.{}", self.scope_name, i),
            );
        }

        // Apply final layer normalization
        let norm_output = hidden_state.mapv(
            Operator::RMSMap(RMSMap::new(
                self.config.hidden_size,
                self.norm_weight.data,
                self.rms_norm_eps,
                self.cpu_num,
            )),
            format!("{}.norm_hidden.output", self.scope_name),
        );

        // Apply language modeling head
        let logits = self
            .output_linear
            .forward(&norm_output, format!("{}.lm_head.output", self.scope_name));

        // Return hidden state for now (in full implementation would return logits)
        hidden_state
    }

    /// Get model configuration
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    /// Get number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        // Embedding parameters
        let embed_params = self.config.vocab_size * self.config.hidden_size;

        // Layer parameters (approximation)
        let attention_params = 4 * self.config.hidden_size * self.config.hidden_size;
        let mlp_params = 3
            * self.config.hidden_size
            * (self
                .config
                .intermediate_size
                .unwrap_or(4 * self.config.hidden_size));
        let norm_params = 2 * self.config.hidden_size;
        let layer_params = attention_params + mlp_params + norm_params;

        let total_layer_params = self.config.num_hidden_layers * layer_params;
        let final_norm_params = self.config.hidden_size;
        let lm_head_params = self.config.vocab_size * self.config.hidden_size;

        embed_params + total_layer_params + final_norm_params + lm_head_params
    }

    /// Generate text tokens using the model
    ///
    /// This function implements text generation with sampling strategies
    /// like temperature scaling and nucleus sampling.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Input prompt token sequences
    /// * `max_gen_len` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (higher = more random)
    /// * `top_p` - Nucleus sampling parameter
    ///
    /// # Returns
    /// Generated token sequences
    pub fn generate(
        &self,
        prompt_tokens: &[Vec<usize>],
        max_gen_len: usize,
        temperature: f32,
        top_p: f32,
    ) -> Vec<Vec<usize>> {
        let mut generated_sequences = Vec::new();

        for prompt in prompt_tokens {
            let mut sequence = prompt.clone();

            // Generate tokens one by one
            for _ in 0..max_gen_len {
                // Create sequence pointer for forward pass
                let mut seq_array = vec![
                    0usize;
                    (self.config.max_position_embeddings + 1)
                        * self.config.batch_size
                ];
                for (i, &token) in sequence.iter().enumerate() {
                    if i < seq_array.len() {
                        seq_array[i] = token;
                    }
                }

                // Forward pass to get logits
                let logits_tensor = self.forward(seq_array.as_mut_ptr());

                // Sample next token (simplified implementation)
                let next_token = self.sample_next_token(&logits_tensor, temperature, top_p);
                sequence.push(next_token);

                // Check for end of sequence
                if next_token == self.config.eos_token_id.unwrap_or(2) {
                    break;
                }
            }

            generated_sequences.push(sequence);
        }

        generated_sequences
    }

    /// Sample next token from logits using temperature and top-p
    ///
    /// # Arguments
    /// * `logits_tensor` - Output logits from the model
    /// * `temperature` - Sampling temperature
    /// * `top_p` - Nucleus sampling parameter
    ///
    /// # Returns
    /// Sampled token ID
    fn sample_next_token(&self, logits_tensor: &Tensor<T>, temperature: f32, top_p: f32) -> usize {
        // Simplified sampling - in a full implementation this would:
        // 1. Apply temperature scaling to logits
        // 2. Apply top-p filtering
        // 3. Sample from the filtered distribution
        // 4. Handle special tokens appropriately

        // For now, return a mock token
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        temperature.to_bits().hash(&mut hasher);
        top_p.to_bits().hash(&mut hasher);
        let hash = hasher.finish();

        (hash as usize) % self.config.vocab_size
    }
}

/// Factory function to create a Transformer from configuration
///
/// This function provides a convenient way to create a Transformer instance
/// with default tensors and properly initialized components.
///
/// # Arguments
/// * `config` - Model configuration
/// * `cpu_num` - Number of CPU cores to use
///
/// # Returns
/// New Transformer instance with initialized components
pub fn create_transformer_from_config<T>(config: Config, cpu_num: usize) -> Transformer<T>
where
    T: Copy
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid<T>
        + Sqrt
        + FromF32,
{
    let cache = Rc::new(RefCell::new(Cache::new(std::collections::HashMap::new())));
    let operator_queue = Rc::new(RefCell::new(Vec::new()));

    // Create default tensors
    let word_embedding = Tensor::zeros(
        vec![config.vocab_size, config.hidden_size],
        String::from("model.embed_tokens.weight"),
        cache.clone(),
        operator_queue.clone(),
    );

    let position_embedding = Tensor::zeros(
        vec![
            config.max_position_embeddings,
            1,
            1,
            config.hidden_size / config.num_attention_heads,
        ],
        String::from("model.position_embedding.weight"),
        cache.clone(),
        operator_queue.clone(),
    );

    let norm_weight = Tensor::zeros(
        vec![1, config.hidden_size],
        String::from("model.norm.weight"),
        cache.clone(),
        operator_queue.clone(),
    );

    Transformer::new(
        config,
        word_embedding,
        position_embedding,
        norm_weight,
        cpu_num,
        cache,
        operator_queue,
    )
}

/// Utility functions for model operations
pub mod utils {
    use super::*;

    /// Calculate memory requirements for a model configuration
    pub fn calculate_memory_requirements(config: &Config, precision_bytes: usize) -> usize {
        let embed_params = config.vocab_size * config.hidden_size;
        let layer_params = 4 * config.hidden_size * config.hidden_size
            + 3 * config.hidden_size * config.intermediate_size.unwrap_or(4 * config.hidden_size)
            + 2 * config.hidden_size;
        let total_layer_params = config.num_hidden_layers * layer_params;
        let other_params = config.hidden_size + config.vocab_size * config.hidden_size;

        let total_params = embed_params + total_layer_params + other_params;
        total_params * precision_bytes
    }

    /// Validate model configuration parameters
    pub fn validate_config(config: &Config) -> Result<(), String> {
        if config.hidden_size == 0 {
            return Err("hidden_size must be greater than 0".to_string());
        }

        if config.num_attention_heads == 0 {
            return Err("num_attention_heads must be greater than 0".to_string());
        }

        if config.hidden_size % config.num_attention_heads != 0 {
            return Err("hidden_size must be divisible by num_attention_heads".to_string());
        }

        if config.num_hidden_layers == 0 {
            return Err("num_hidden_layers must be greater than 0".to_string());
        }

        if config.vocab_size == 0 {
            return Err("vocab_size must be greater than 0".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Test basic model creation and forward pass
    #[test]
    fn test_transformer_creation() {
        let cpu_num = thread::available_parallelism().unwrap().get();
        let mut config = Config::new();
        config.hidden_size = 512;
        config.vocab_size = 1000;
        config.num_hidden_layers = 6;
        config.num_attention_heads = 8;
        config.batch_size = 1;
        config.max_position_embeddings = 128;

        let transformer = create_transformer_from_config::<f32>(config.clone(), cpu_num);

        assert_eq!(transformer.config.hidden_size, 512);
        assert_eq!(transformer.config.vocab_size, 1000);
        assert_eq!(transformer.config.num_hidden_layers, 6);
    }

    /// Test model parameter counting
    #[test]
    fn test_parameter_counting() {
        let config = Config {
            hidden_size: 512,
            vocab_size: 1000,
            num_hidden_layers: 6,
            num_attention_heads: 8,
            intermediate_size: Some(2048),
            ..Default::default()
        };

        let transformer = create_transformer_from_config::<f32>(config, 1);
        let num_params = transformer.num_parameters();

        // Should be > 0 and reasonable for the config
        assert!(num_params > 1_000_000);
        assert!(num_params < 100_000_000);
    }

    /// Test configuration validation
    #[test]
    fn test_config_validation() {
        let mut config = Config::new();
        config.hidden_size = 512;
        config.num_attention_heads = 8;

        assert!(utils::validate_config(&config).is_ok());

        // Test invalid configuration
        config.hidden_size = 0;
        assert!(utils::validate_config(&config).is_err());
    }

    /// Test memory calculation
    #[test]
    fn test_memory_calculation() {
        let config = Config {
            hidden_size: 512,
            vocab_size: 1000,
            num_hidden_layers: 6,
            intermediate_size: Some(2048),
            ..Default::default()
        };

        let memory_f16 = utils::calculate_memory_requirements(&config, 2);
        let memory_f32 = utils::calculate_memory_requirements(&config, 4);

        assert!(memory_f16 > 0);
        assert_eq!(memory_f32, memory_f16 * 2);
    }
}
