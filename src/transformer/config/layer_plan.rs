use serde::{Deserialize, Serialize};

use super::attention_kind::AttentionKind;
use super::ffn_kind::{FfnKind, FfnResolveParams};
use super::router_scoring::RouterScoringKind;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerPlan {
    pub attention: AttentionKind,
    pub ffn: FfnKind,
}

impl LayerPlan {
    pub(crate) fn build_stack(
        num_hidden_layers: usize,
        use_sliding_window: bool,
        max_window_layers: usize,
        layer_types: Option<&[String]>,
        ffn_params: &FfnResolveParams<'_>,
        router_scoring: &RouterScoringKind,
        use_routing_bias: bool,
    ) -> Vec<LayerPlan> {
        (0..num_hidden_layers)
            .map(|layer_idx| LayerPlan {
                attention: AttentionKind::for_layer(
                    layer_types,
                    layer_idx,
                    use_sliding_window,
                    max_window_layers,
                ),
                ffn: FfnKind::for_layer(ffn_params, layer_idx, router_scoring, use_routing_bias),
            })
            .collect()
    }
}
