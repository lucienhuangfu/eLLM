use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    Full,
    SlidingWindow,
    Linear,
}

impl AttentionKind {
    pub(crate) fn for_layer(
        layer_types: Option<&[String]>,
        layer_idx: usize,
        use_sliding_window: bool,
        max_window_layers: usize,
    ) -> Self {
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

        if use_sliding_window && layer_idx < max_window_layers {
            return AttentionKind::SlidingWindow;
        }

        AttentionKind::Full
    }
}
