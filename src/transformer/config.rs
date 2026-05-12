use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamily {
    Qwen,
    Llama,
    Mixtral,
    MiniMax,
    MiniMaxM2,
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    Full,
    SlidingWindow,
    Linear,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FfnKind {
    Dense {
        intermediate_size: usize,
    },
    SparseMoe {
        intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        router_scoring: RouterScoringKind,
        use_routing_bias: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouterScoringKind {
    Softmax,
    Sigmoid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerPlan {
    pub attention: AttentionKind,
    pub ffn: FfnKind,
}

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
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub intermediate_size: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct HfConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    attention_dropout: f32,
    #[serde(default = "default_decoder_sparse_step")]
    decoder_sparse_step: usize,
    #[serde(default)]
    eos_token_id: usize,
    head_dim: Option<usize>,
    #[serde(default)]
    hidden_act: String,
    #[serde(default)]
    hidden_size: usize,
    #[serde(default)]
    initializer_range: f32,
    intermediate_size: Option<usize>,
    #[serde(default)]
    max_position_embeddings: usize,
    max_window_layers: Option<usize>,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
    #[serde(default)]
    model_type: String,
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    num_attention_heads: usize,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    num_hidden_layers: usize,
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    output_router_logits: bool,
    #[serde(default)]
    qkv_bias: bool,
    #[serde(default)]
    rms_norm_eps: f32,
    rope_scaling: Option<HashMap<String, Value>>,
    rope_theta: Option<usize>,
    rotary_dim: Option<usize>,
    scoring_func: Option<String>,
    #[serde(default)]
    router_aux_loss_coef: f32,
    shared_experts_intermediate_size: Option<usize>,
    sliding_window: Option<usize>,
    use_routing_bias: Option<bool>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    torch_dtype: String,
    #[serde(default)]
    transformers_version: String,
    #[serde(default)]
    use_cache: bool,
    #[serde(default)]
    use_qk_norm: bool,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    vocab_size: usize,
    layer_types: Option<Vec<String>>,
}

fn default_decoder_sparse_step() -> usize {
    1
}

impl Config {
    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let raw: HfConfig = serde_json::from_reader(reader)?;
        Ok(Self::from_hf(raw))
    }

    fn from_hf(raw: HfConfig) -> Self {
        let family = resolve_family(&raw.model_type);
        let head_dim = raw
            .head_dim
            .unwrap_or_else(|| raw.hidden_size / raw.num_attention_heads.max(1));
        let num_key_value_heads = raw
            .num_key_value_heads
            .unwrap_or(raw.num_attention_heads.max(1));
        let intermediate_size = raw
            .intermediate_size
            .unwrap_or_else(|| raw.moe_intermediate_size.unwrap_or(raw.hidden_size));
        let moe_intermediate_size = raw.moe_intermediate_size.unwrap_or(intermediate_size);
        let num_experts = raw.num_experts.unwrap_or(0);
        let num_experts_per_tok = raw.num_experts_per_tok.unwrap_or(0);
        let max_window_layers = raw.max_window_layers.unwrap_or(raw.num_hidden_layers);
        let router_scoring =
            resolve_router_scoring(raw.scoring_func.as_deref(), resolve_family(&raw.model_type));
        let use_routing_bias = raw
            .use_routing_bias
            .unwrap_or(matches!(family, ModelFamily::MiniMaxM2));
        let decoder_sparse_step = raw.decoder_sparse_step.max(1);

        let mut config = Self {
            family,
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta.unwrap_or(10000),
            rotary_dim: raw.rotary_dim.unwrap_or(head_dim),
            tie_word_embeddings: raw.tie_word_embeddings,
            layers: Vec::with_capacity(raw.num_hidden_layers),
            qkv_bias: raw.qkv_bias,
            use_qk_norm: raw.use_qk_norm,
            rope_scaling: raw.rope_scaling,
            eos_token_id: raw.eos_token_id,
            max_window_layers,
            use_sliding_window: raw.use_sliding_window,
            sliding_window: raw.sliding_window,
            intermediate_size,
        };

        // Extract layer_types before borrowing mlp_only_layers
        let layer_types = raw.layer_types;

        // Resolve layer plans using raw HfConfig values (not stored in ResolvedConfig)
        let ffn_params = FfnResolveParams {
            mlp_only_layers: &raw.mlp_only_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob: raw.norm_topk_prob,
            decoder_sparse_step,
            intermediate_size,
        };

        config.layers = (0..config.num_hidden_layers)
            .map(|layer_idx| LayerPlan {
                attention: resolve_attention_kind(&config, layer_types.as_deref(), layer_idx),
                ffn: resolve_ffn_kind(&ffn_params, layer_idx, &router_scoring, use_routing_bias),
            })
            .collect();

        config
    }
}

fn resolve_family(model_type: &str) -> ModelFamily {
    let model_type = model_type.to_ascii_lowercase();
    match model_type.as_str() {
        "qwen2" | "qwen2_moe" | "qwen3" | "qwen3_moe" => ModelFamily::Qwen,
        "llama" => ModelFamily::Llama,
        "mixtral" => ModelFamily::Mixtral,
        "minimax" => ModelFamily::MiniMax,
        "minimax_m2" | "minimax-m2" | "minimax_m2.5" | "minimax-m2.5" => ModelFamily::MiniMaxM2,
        _ => ModelFamily::Unknown(model_type),
    }
}

fn resolve_router_scoring(scoring_func: Option<&str>, family: ModelFamily) -> RouterScoringKind {
    match scoring_func.map(|s| s.to_ascii_lowercase()) {
        Some(scoring) if scoring == "sigmoid" => RouterScoringKind::Sigmoid,
        Some(scoring) if scoring == "softmax" => RouterScoringKind::Softmax,
        _ => match family {
            ModelFamily::MiniMaxM2 => RouterScoringKind::Sigmoid,
            _ => RouterScoringKind::Softmax,
        },
    }
}

fn resolve_attention_kind(
    config: &Config,
    layer_types: Option<&[String]>,
    layer_idx: usize,
) -> AttentionKind {
    if let Some(layer_types) = layer_types {
        if let Some(layer_type) = layer_types.get(layer_idx) {
            let layer_type = layer_type.to_ascii_lowercase();
            if layer_type.contains("linear") {
                return AttentionKind::Linear;
            }
            if layer_type.contains("sliding") || layer_type.contains("window") {
                return AttentionKind::SlidingWindow;
            }
        }
    }

    if config.use_sliding_window && layer_idx < config.max_window_layers {
        return AttentionKind::SlidingWindow;
    }

    AttentionKind::Full
}

struct FfnResolveParams<'a> {
    mlp_only_layers: &'a [usize],
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    norm_topk_prob: bool,
    decoder_sparse_step: usize,
    intermediate_size: usize,
}

fn resolve_ffn_kind(
    params: &FfnResolveParams,
    layer_idx: usize,
    router_scoring: &RouterScoringKind,
    use_routing_bias: bool,
) -> FfnKind {
    if params.mlp_only_layers.contains(&layer_idx) || params.num_experts == 0 {
        return FfnKind::Dense {
            intermediate_size: params.intermediate_size,
        };
    }

    if (layer_idx + 1) % params.decoder_sparse_step == 0 {
        return FfnKind::SparseMoe {
            intermediate_size: params.moe_intermediate_size,
            num_experts: params.num_experts,
            num_experts_per_tok: params.num_experts_per_tok,
            norm_topk_prob: params.norm_topk_prob,
            router_scoring: router_scoring.clone(),
            use_routing_bias,
        };
    }

    FfnKind::Dense {
        intermediate_size: params.intermediate_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_file() {
        let config = Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json");
        match config {
            Ok(cfg) => {
                println!("{:?}", cfg.family);
                assert_eq!(cfg.layers.len(), cfg.num_hidden_layers);
            }
            Err(e) => println!("Error loading config: {}", e),
        }
    }
}
