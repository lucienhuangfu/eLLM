use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

/// Raw Hugging Face `config.json` shape: strings, numbers, and options only — no runtime enums.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct HfConfig {
    #[serde(default)]
    pub(crate) architectures: Vec<String>,
    #[serde(default)]
    pub(crate) attention_dropout: f32,
    #[serde(default = "default_decoder_sparse_step")]
    pub(crate) decoder_sparse_step: usize,
    #[serde(default)]
    pub(crate) eos_token_id: usize,
    pub(crate) head_dim: Option<usize>,
    #[serde(default)]
    pub(crate) hidden_act: String,
    #[serde(default)]
    pub(crate) hidden_size: usize,
    #[serde(default)]
    pub(crate) initializer_range: f32,
    pub(crate) intermediate_size: Option<usize>,
    #[serde(default)]
    pub(crate) max_position_embeddings: usize,
    pub(crate) max_window_layers: Option<usize>,
    #[serde(default)]
    pub(crate) mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub(crate) model_type: String,
    pub(crate) moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub(crate) norm_topk_prob: bool,
    #[serde(default)]
    pub(crate) num_attention_heads: usize,
    pub(crate) num_experts: Option<usize>,
    pub(crate) num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub(crate) output_router_logits: bool,
    #[serde(default)]
    pub(crate) qkv_bias: bool,
    #[serde(default)]
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_scaling: Option<HashMap<String, Value>>,
    pub(crate) rope_theta: Option<usize>,
    pub(crate) rotary_dim: Option<usize>,
    pub(crate) scoring_func: Option<String>,
    #[serde(default)]
    pub(crate) router_aux_loss_coef: f32,
    pub(crate) shared_experts_intermediate_size: Option<usize>,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) use_routing_bias: Option<bool>,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
    #[serde(default)]
    pub(crate) torch_dtype: String,
    #[serde(default)]
    pub(crate) transformers_version: String,
    #[serde(default)]
    pub(crate) use_cache: bool,
    #[serde(default)]
    pub(crate) use_qk_norm: bool,
    #[serde(default)]
    pub(crate) use_sliding_window: bool,
    #[serde(default)]
    pub(crate) vocab_size: usize,
    pub(crate) layer_types: Option<Vec<String>>,
}

fn default_decoder_sparse_step() -> usize {
    1
}

impl HfConfig {
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_reader(reader)?)
    }

    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        Self::from_reader(BufReader::new(file))
    }
}
