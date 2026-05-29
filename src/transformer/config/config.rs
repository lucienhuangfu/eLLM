use std::path::Path;

use serde_json::Value;
use std::collections::HashMap;

use crate::runtime::{generation_config::EosTokenIds, HfConfig};

use super::ffn_kind::FfnResolveParams;
use super::layer_plan::LayerPlan;
use super::model_family::ModelFamily;
use super::router_scoring::RouterScoringKind;

#[derive(Debug, Clone)]
pub struct Config {
    pub family: ModelFamily,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: usize,
    pub rotary_dim: usize,
    pub tie_word_embeddings: bool,
    pub layers: Vec<LayerPlan>,
    pub qkv_bias: bool,
    pub use_qk_norm: bool,
    pub rope_scaling: Option<HashMap<String, Value>>,
    pub eos_token_id: usize,
    pub eos_token_ids: EosTokenIds,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub intermediate_size: usize,
}

impl Config {
    pub fn from_hf(hf: HfConfig) -> Self {
        let family = ModelFamily::parse(&hf.model_type);
        let head_dim = hf
            .head_dim
            .unwrap_or_else(|| hf.hidden_size / hf.num_attention_heads.max(1));
        let num_key_value_heads = hf
            .num_key_value_heads
            .unwrap_or(hf.num_attention_heads.max(1));
        let intermediate_size = hf
            .intermediate_size
            .unwrap_or_else(|| hf.moe_intermediate_size.unwrap_or(hf.hidden_size));
        let moe_intermediate_size = hf.moe_intermediate_size.unwrap_or(intermediate_size);
        let num_experts = hf.num_experts.unwrap_or(0);
        let num_experts_per_tok = hf.num_experts_per_tok.unwrap_or(0);
        let max_window_layers = hf.max_window_layers.unwrap_or(hf.num_hidden_layers);
        let router_scoring = RouterScoringKind::from_hf(hf.scoring_func.as_deref(), family.clone());
        let use_routing_bias = hf
            .use_routing_bias
            .unwrap_or(matches!(family, ModelFamily::MiniMaxM2));
        let decoder_sparse_step = hf.decoder_sparse_step.max(1);
        let use_qk_norm = hf.use_qk_norm || matches!(hf.model_type.as_str(), "qwen3" | "qwen3_moe");

        let layer_types = hf.layer_types;

        let ffn_params = FfnResolveParams {
            mlp_only_layers: &hf.mlp_only_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob: hf.norm_topk_prob,
            decoder_sparse_step,
            intermediate_size,
        };

        let layers = LayerPlan::build_stack(
            hf.num_hidden_layers,
            hf.use_sliding_window,
            max_window_layers,
            layer_types.as_deref(),
            &ffn_params,
            &router_scoring,
            use_routing_bias,
        );

        let eos_token_ids = hf.eos_token_id.to_eos_token_ids();
        let eos_token_id = hf.eos_token_id.first().unwrap_or(0);

        Self {
            family,
            vocab_size: hf.vocab_size,
            hidden_size: hf.hidden_size,
            num_hidden_layers: hf.num_hidden_layers,
            num_attention_heads: hf.num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings: hf.max_position_embeddings,
            rms_norm_eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta.unwrap_or(10000),
            rotary_dim: hf.rotary_dim.unwrap_or(head_dim),
            tie_word_embeddings: hf.tie_word_embeddings,
            layers,
            qkv_bias: hf.qkv_bias,
            use_qk_norm,
            rope_scaling: hf.rope_scaling,
            eos_token_id,
            eos_token_ids,
            max_window_layers,
            use_sliding_window: hf.use_sliding_window,
            sliding_window: hf.sliding_window,
            intermediate_size,
        }
    }

    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::from_hf(HfConfig::load_from_file(filename)?))
    }
}

#[cfg(test)]
mod tests {
    use crate::transformer::config::{Config, HfConfig};

    #[test]
    fn test_from_file() {
        let path = r"models/Qwen3-Coder-30B-A3B-Instruct/config.json";
        let config = match HfConfig::load_from_file(path) {
            Ok(hf) => Config::from_hf(hf),
            Err(e) => {
                println!("Error loading config: {}", e);
                return;
            }
        };
        println!("{:?}", config.family);
        assert_eq!(config.layers.len(), config.num_hidden_layers);
    }
}
