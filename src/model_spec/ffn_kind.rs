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
