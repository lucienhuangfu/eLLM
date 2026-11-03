use serde::{Deserialize, Serialize};

use crate::transformer::mlp::{FfnKind, RouterScoringKind};

/// Lazy per-layer blueprint resolver: global parameters are stored once and
/// `ffn(i)` is computed on demand, replacing a previously materialized
/// per-layer vector (no per-layer cloning of global MoE params). Attention
/// typing is resolved by the runtime block from the raw `layer_type(i)` string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub(crate) num_hidden_layers: usize,
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

    /// 第 layer_idx 层的原始 layer_type 字符串；缺省 None。
    /// 类型分派由 `AttentionBlock::new` 按字符串完成。
    pub fn layer_type(&self, layer_idx: usize) -> Option<&str> {
        self.layer_types
            .as_ref()
            .and_then(|types| types.get(layer_idx))
            .map(|s| s.as_str())
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
