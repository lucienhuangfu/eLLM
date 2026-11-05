//! 视觉编码器配置与顶层多模态装配用到的常量：`config.json` 的 `vision_config`
//! 子对象，以及四个视觉特殊 token id。
//!
//! 对应 `scripts/qwen3_5/configuration_qwen3_5.py` 的 `Qwen3_5VisionConfig` 与
//! `Qwen3_5Config` 的视觉部分。形状沿用本仓库既有的 `VisionConfig`：`Default` 即
//! py dataclass 默认值，配合结构级 `#[serde(default)]`，缺键自动落到该默认，
//! 因而不再需要 `Option` 双层与逐字段 `unwrap_or`。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// py `Qwen3_5Config` 的视觉特殊 token id 默认值。
pub const DEFAULT_IMAGE_TOKEN_ID: usize = 248056;
pub const DEFAULT_VIDEO_TOKEN_ID: usize = 248057;
pub const DEFAULT_VISION_START_TOKEN_ID: usize = 248053;
pub const DEFAULT_VISION_END_TOKEN_ID: usize = 248054;

/// 视觉编码器（ViT）超参，对应 `Qwen3_5VisionConfig`。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub hidden_act: String,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub initializer_range: f32,
    /// 可为显式 `null`，故保留 `Option`。
    pub rope_parameters: Option<HashMap<String, Value>>,
}

/// `Default` 即 py dataclass 的类默认值；`#[serde(default)]` 在键缺失时取的就是这些值
/// （对应 `Qwen3_5VisionConfig` 里 `depth: int = 27` 那一串）。
impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            depth: 27,
            hidden_size: 1152,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            intermediate_size: 4304,
            num_heads: 16,
            in_channels: 3,
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: 3584,
            num_position_embeddings: 2304,
            initializer_range: 0.02,
            rope_parameters: None,
        }
    }
}

/// 视觉特殊 token id，对应 `Qwen3_5Config` 的四个顶层字段。
#[derive(Debug, Clone, Copy)]
pub struct VisionTokens {
    pub image_token_id: usize,
    pub video_token_id: usize,
    pub vision_start_token_id: usize,
    pub vision_end_token_id: usize,
}

impl Default for VisionTokens {
    fn default() -> Self {
        Self {
            image_token_id: DEFAULT_IMAGE_TOKEN_ID,
            video_token_id: DEFAULT_VIDEO_TOKEN_ID,
            vision_start_token_id: DEFAULT_VISION_START_TOKEN_ID,
            vision_end_token_id: DEFAULT_VISION_END_TOKEN_ID,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_vision_keys_fall_back_to_py_defaults() {
        let cfg: VisionConfig =
            serde_json::from_str(r#"{"depth": 24, "hidden_size": 1024}"#).unwrap();
        assert_eq!(cfg.depth, 24);
        assert_eq!(cfg.hidden_size, 1024);
        // 未给出的键取 py 默认。
        assert_eq!(cfg.intermediate_size, 4304);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.out_hidden_size, 3584);
        assert_eq!(cfg.hidden_act, "gelu_pytorch_tanh");
    }

    #[test]
    fn vision_tokens_default_to_py_ids() {
        let t = VisionTokens::default();
        assert_eq!(t.image_token_id, DEFAULT_IMAGE_TOKEN_ID);
        assert_eq!(t.vision_start_token_id, DEFAULT_VISION_START_TOKEN_ID);
    }
}
