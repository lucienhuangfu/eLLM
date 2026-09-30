use serde::{Deserialize, Serialize};

use super::attention_kind::AttentionKind;
use super::ffn_kind::FfnKind;
use super::router_scoring::RouterScoringKind;

/// Lazy per-layer blueprint resolver: global parameters are stored once and
/// `attention(i)` / `ffn(i)` are computed on demand, replacing a previously
/// materialized per-layer vector (no per-layer cloning of global MoE params).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub(crate) num_hidden_layers: usize,
    pub(crate) use_sliding_window: bool,
    pub(crate) max_window_layers: usize,
    pub(crate) layer_types: Option<Vec<String>>,
    pub(crate) mlp_only_layers: Vec<usize>,
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) norm_topk_prob: bool,
    pub(crate) decoder_sparse_step: usize,
    pub(crate) router_scoring: RouterScoringKind,
    pub(crate) use_routing_bias: bool,
}

impl LayerSpec {
    pub fn len(&self) -> usize {
        self.num_hidden_layers
    }

    pub fn is_empty(&self) -> bool {
        self.num_hidden_layers == 0
    }

    pub fn attention(&self, layer_idx: usize) -> AttentionKind {
        if let Some(layer_types) = &self.layer_types {
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

        if self.use_sliding_window && layer_idx < self.max_window_layers {
            return AttentionKind::SlidingWindow;
        }

        AttentionKind::Full
    }

    pub fn ffn(&self, layer_idx: usize) -> FfnKind {
        if self.mlp_only_layers.contains(&layer_idx) || self.num_experts == 0 {
            return FfnKind::Dense {
                intermediate_size: self.intermediate_size,
            };
        }

        if (layer_idx + 1) % self.decoder_sparse_step == 0 {
            return FfnKind::SparseMoe {
                intermediate_size: self.moe_intermediate_size,
                num_experts: self.num_experts,
                num_experts_per_tok: self.num_experts_per_tok,
                norm_topk_prob: self.norm_topk_prob,
                router_scoring: self.router_scoring.clone(),
                use_routing_bias: self.use_routing_bias,
            };
        }

        FfnKind::Dense {
            intermediate_size: self.intermediate_size,
        }
    }
}
