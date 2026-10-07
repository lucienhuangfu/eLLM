//! Rust translation of `scripts/qwen3_5/configuration_qwen3_5.py`.
//! `scripts/qwen3_5/configuration_qwen3_5.py` 的 Rust 翻译。
//!
//! 顶层多模态配置：文本主干复用通用的 `model_spec::TextConfig`（经 `profile()`
//! 注入 qwen3_5 默认值），本模块只保留 qwen3_5 真正特有的部分——视觉编码器
//! `VisionConfig` 与顶层 `Config`（text + vision 子配置 + 视觉特殊 token）。
//!
//! py 类对应：
//! - `Qwen3_5TextConfig`   -> `model_spec::TextConfig`（默认值由 `profile()` 提供）
//! - `Qwen3_5VisionConfig` -> [`VisionConfig`]
//! - `Qwen3_5Config`       -> [`Config`]（顶层，含 text/vision 子配置与视觉特殊 token）
//!
//! 分层沿用 `model_spec::TextConfig`：`HfConfig` / `HfVisionConfig` 是“原始 JSON 层”
//! （字段多为 `Option`），py 端 dataclass 默认值与 `__post_init__` 派生逻辑在
//! `profile()` 与 `from_hf` 中应用。

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::config::{HfConfig, HfVisionConfig};
use crate::model_spec::{FamilyProfile, LinearDefaults, TextConfig};

// ── Qwen3_5TextConfig dataclass 默认值（config.json 缺字段时兜底，忠实 py）──
// 这些默认值经 profile() 注入通用 TextConfig::from_hf。
const DEFAULT_LINEAR_CONV_KERNEL_DIM: usize = 4;
const DEFAULT_LINEAR_KEY_HEAD_DIM: usize = 128;
const DEFAULT_LINEAR_VALUE_HEAD_DIM: usize = 128;
const DEFAULT_LINEAR_NUM_KEY_HEADS: usize = 16;
const DEFAULT_LINEAR_NUM_VALUE_HEADS: usize = 32;
// py `__post_init__`: kwargs.setdefault("partial_rotary_factor", 0.25)。
const DEFAULT_PARTIAL_ROTARY_FACTOR: f32 = 0.25;
// py `__post_init__`: kwargs.pop("full_attention_interval", 4)。
const DEFAULT_FULL_ATTENTION_INTERVAL: usize = 4;

// ── Qwen3_5Config 视觉特殊 token id 默认值（忠实 py）──
const DEFAULT_IMAGE_TOKEN_ID: usize = 248056;
const DEFAULT_VIDEO_TOKEN_ID: usize = 248057;
const DEFAULT_VISION_START_TOKEN_ID: usize = 248053;
const DEFAULT_VISION_END_TOKEN_ID: usize = 248054;

/// qwen3_5 的 family 默认值，注入通用 `TextConfig::from_hf`。
///
/// 忠实 py `Qwen3_5TextConfig.__post_init__`：partial_rotary_factor=0.25
/// （rotary_dim = head_dim × 0.25，py head_dim=256 → 64）、full_attention_interval=4
/// （layer_types 缺省时每隔 4 层放一个 full_attention，其余 linear_attention）、
/// GatedDeltaNet 线性注意力块默认维度、qwen3 系默认启用 qk-norm。
pub fn profile() -> FamilyProfile {
    FamilyProfile {
        partial_rotary_factor: DEFAULT_PARTIAL_ROTARY_FACTOR,
        full_attention_interval: Some(DEFAULT_FULL_ATTENTION_INTERVAL),
        linear_defaults: LinearDefaults {
            num_k: DEFAULT_LINEAR_NUM_KEY_HEADS,
            num_v: DEFAULT_LINEAR_NUM_VALUE_HEADS,
            key_dim: DEFAULT_LINEAR_KEY_HEAD_DIM,
            value_dim: DEFAULT_LINEAR_VALUE_HEAD_DIM,
            conv_kernel: DEFAULT_LINEAR_CONV_KERNEL_DIM,
        },
        qk_norm: true,
        routing_bias: false,
    }
}

/// 视觉编码器配置，对应 `Qwen3_5VisionConfig`。
///
/// `Default` 即 py dataclass 默认值；`from_hf` 对每个字段做 `unwrap_or(默认)`，
/// 因此 config.json 里缺失的字段会回退到 py 默认，显式给出的字段则以 JSON 为准。
#[derive(Debug, Clone)]
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
    pub rope_parameters: Option<HashMap<String, Value>>,
}

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

impl VisionConfig {
    pub fn from_hf(hf: &HfVisionConfig) -> Self {
        let d = Self::default();
        Self {
            depth: hf.depth.unwrap_or(d.depth),
            hidden_size: hf.hidden_size.unwrap_or(d.hidden_size),
            hidden_act: hf.hidden_act.clone().unwrap_or(d.hidden_act),
            intermediate_size: hf.intermediate_size.unwrap_or(d.intermediate_size),
            num_heads: hf.num_heads.unwrap_or(d.num_heads),
            in_channels: hf.in_channels.unwrap_or(d.in_channels),
            patch_size: hf.patch_size.unwrap_or(d.patch_size),
            spatial_merge_size: hf.spatial_merge_size.unwrap_or(d.spatial_merge_size),
            temporal_patch_size: hf.temporal_patch_size.unwrap_or(d.temporal_patch_size),
            out_hidden_size: hf.out_hidden_size.unwrap_or(d.out_hidden_size),
            num_position_embeddings: hf
                .num_position_embeddings
                .unwrap_or(d.num_position_embeddings),
            initializer_range: hf.initializer_range.unwrap_or(d.initializer_range),
            rope_parameters: hf.rope_parameters.clone(),
        }
    }
}

/// 顶层多模态配置，对应 `Qwen3_5Config`：文本 + 视觉两个子配置，外加视觉特殊 token id。
///
/// py `__post_init__` 中 text_config / vision_config 缺省会实例化默认子配置——
/// 这里 `from_hf` 同样处理：嵌套子配置缺失时，text 回退到顶层扁平字段，vision 回退到
/// `VisionConfig::default()`。文本子配置复用通用 `model_spec::TextConfig`，其 qwen3_5
/// 默认值由 `profile()` 注入。
#[derive(Debug, Clone)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: usize,
    pub video_token_id: usize,
    pub vision_start_token_id: usize,
    pub vision_end_token_id: usize,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn from_hf(hf: HfConfig) -> Self {
        // 优先嵌套 text_config；缺失时回退顶层扁平字段（兼容非标准 dump）。
        let text_hf: &HfConfig = hf.text_config.as_deref().unwrap_or(&hf);
        let text_config = TextConfig::from_hf(text_hf, &profile());
        // vision_config 缺失 → py 默认（对应 `elif self.vision_config is None`）。
        let vision_config = hf
            .vision_config
            .as_ref()
            .map(VisionConfig::from_hf)
            .unwrap_or_default();

        Self {
            text_config,
            vision_config,
            image_token_id: hf.image_token_id.unwrap_or(DEFAULT_IMAGE_TOKEN_ID),
            video_token_id: hf.video_token_id.unwrap_or(DEFAULT_VIDEO_TOKEN_ID),
            vision_start_token_id: hf
                .vision_start_token_id
                .unwrap_or(DEFAULT_VISION_START_TOKEN_ID),
            vision_end_token_id: hf
                .vision_end_token_id
                .unwrap_or(DEFAULT_VISION_END_TOKEN_ID),
            tie_word_embeddings: hf.tie_word_embeddings,
        }
    }

    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::from_hf(HfConfig::load_from_file(filename)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_spec::{AttentionKind, FfnKind, ModelName};

    #[test]
    fn text_config_derives_rotary_dim_and_layer_types() {
        // 嵌套 config：head_dim=256、partial_rotary_factor 缺省(profile 0.25)、num_hidden_layers=8。
        let json = r#"{
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 8,
                "hidden_size": 4096,
                "num_attention_heads": 16,
                "head_dim": 256
            }
        }"#;
        let hf: HfConfig = serde_json::from_str(json).unwrap();
        let text = Config::from_hf(hf).text_config;

        assert!(matches!(text.family, ModelName::Qwen3_5));
        // rotary_dim = 256 × 0.25 = 64。
        assert_eq!(text.rotary_dim, 64);
        assert_eq!(text.head_dim, 256);
        // qk-norm 对 qwen3_5 默认开启。
        assert!(text.use_qk_norm);
        // GatedDeltaNet 线性块默认值来自 profile()。
        assert_eq!(text.linear_num_key_heads, 16);
        assert_eq!(text.linear_num_value_heads, 32);
        assert_eq!(text.linear_conv_kernel_dim, 4);

        // layer_types 自动生成（full_attention_interval=4）：索引 3、7 为 full，其余 linear。
        assert_eq!(text.layer_spec.len(), 8);
        assert_eq!(text.layer_spec.attention(0), AttentionKind::Linear);
        assert_eq!(text.layer_spec.attention(2), AttentionKind::Linear);
        assert_eq!(text.layer_spec.attention(3), AttentionKind::Full);
        assert_eq!(text.layer_spec.attention(4), AttentionKind::Linear);
        assert_eq!(text.layer_spec.attention(7), AttentionKind::Full);

        // dense 文本部分：无专家。
        assert!(matches!(text.layer_spec.ffn(0), FfnKind::Dense { .. }));
    }

    #[test]
    fn vision_and_token_ids_fall_back_to_py_defaults() {
        // 仅提供 model_type：text_config / vision_config / token id 全部缺省。
        let json = r#"{ "model_type": "qwen3_5" }"#;
        let hf: HfConfig = serde_json::from_str(json).unwrap();
        let config = Config::from_hf(hf);

        // 视觉子配置回退 py 默认。
        assert_eq!(config.vision_config.depth, 27);
        assert_eq!(config.vision_config.hidden_size, 1152);
        assert_eq!(config.vision_config.out_hidden_size, 3584);
        assert_eq!(config.vision_config.hidden_act, "gelu_pytorch_tanh");

        // 视觉特殊 token id 回退 py 默认。
        assert_eq!(config.image_token_id, DEFAULT_IMAGE_TOKEN_ID);
        assert_eq!(config.video_token_id, DEFAULT_VIDEO_TOKEN_ID);
        assert_eq!(config.vision_start_token_id, DEFAULT_VISION_START_TOKEN_ID);
        assert_eq!(config.vision_end_token_id, DEFAULT_VISION_END_TOKEN_ID);
    }

    #[test]
    fn vision_config_honors_explicit_json_values() {
        let json = r#"{
            "model_type": "qwen3_5",
            "vision_config": {
                "model_type": "qwen3_5_vision",
                "depth": 24,
                "hidden_size": 1024,
                "out_hidden_size": 2048
            }
        }"#;
        let hf: HfConfig = serde_json::from_str(json).unwrap();
        let vision = Config::from_hf(hf).vision_config;

        // 显式给出的以 JSON 为准，未给出的回退默认。
        assert_eq!(vision.depth, 24);
        assert_eq!(vision.hidden_size, 1024);
        assert_eq!(vision.out_hidden_size, 2048);
        assert_eq!(vision.intermediate_size, 4304);
        assert_eq!(vision.num_heads, 16);
    }
}
