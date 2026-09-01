use serde::{Deserialize, Serialize};

use super::router_scoring::RouterScoringKind;

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

pub(crate) struct FfnResolveParams<'a> {
    pub(crate) mlp_only_layers: &'a [usize],
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) norm_topk_prob: bool,
    pub(crate) decoder_sparse_step: usize,
    pub(crate) intermediate_size: usize,
}

impl FfnKind {
    pub(crate) fn for_layer(
        params: &FfnResolveParams<'_>,
        layer_idx: usize,
        router_scoring: &RouterScoringKind,
        use_routing_bias: bool,
    ) -> Self {
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
}
