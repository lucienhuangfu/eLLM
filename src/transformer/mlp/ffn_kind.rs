use serde::{Deserialize, Serialize};

use super::RouterScoringKind;

/// FFN 配置蓝图：每层的 FFN 形态（Dense / SparseMoe）及其超参数，
/// 由 `LayerSpec::ffn(i)` 按需解析，`FfnBlock::new` 消费实例化。
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
