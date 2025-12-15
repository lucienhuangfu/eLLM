use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub architectures: Vec<String>,
    pub attention_dropout: f32,
    pub decoder_sparse_step: usize,
    pub eos_token_id: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub mlp_only_layers: Vec<usize>,
    pub model_type: String,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub output_router_logits: bool,
    pub qkv_bias: bool,
    pub rms_norm_eps: f32,
    pub rope_scaling: Option<HashMap<String, String>>,
    pub rope_theta: usize,
    pub router_aux_loss_coef: f32,
    pub shared_experts_intermediate_size: usize,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub use_qk_norm: bool,
    pub use_sliding_window: bool,
    pub vocab_size: usize,
}

impl Config {
    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let config: Config = serde_json::from_reader(reader)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_file() {
        let config = Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json");
        match config {
            Ok(cfg) => println!("{:?}", cfg),
            Err(e) => println!("Error loading config: {}", e),
        }
    }
}
