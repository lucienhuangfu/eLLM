//! Rust translation of `scripts/qwen3_5/configuration_qwen3_5.py`.
//! `scripts/qwen3_5/configuration_qwen3_5.py` 的 Rust 翻译。
//!
//! 三个 py 配置类一一对应：
//! - `Qwen3_5TextConfig`   -> [`TextConfig`]
//! - `Qwen3_5VisionConfig` -> [`VisionConfig`]
//! - `Qwen3_5Config`       -> [`Config`]（顶层，含 text/vision 子配置与视觉特殊 token）
//!
//! 分层沿用 `model_family::qwen3_moe::model_config`：`HfConfig` / `HfVisionConfig`
//! 是“原始 JSON 层”（字段多为 `Option`），py 端 dataclass 默认值与 `__post_init__`
//! 派生逻辑在此处的 `from_hf` 中应用。

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::config::{HfConfig, HfVisionConfig};
use crate::model_spec::{LayerSpec, ModelName, RouterScoringKind};

// ── Qwen3_5TextConfig dataclass 默认值（config.json 缺字段时兜底，忠实 py）──
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

/// 文本子配置，对应 `Qwen3_5TextConfig`。
///
/// 字段集对齐 `model_family::qwen3_moe::Config`（同为 GatedDeltaNet 混合注意力），
/// 便于未来复用 `DecoderLayer` / `Attention` 的字段消费。相对 qwen3_moe 的差异：
/// - `rotary_dim` 由 `partial_rotary_factor`（默认 0.25）派生：`head_dim × factor`
///   （py head_dim=256 → rotary_dim=64）；
/// - `layer_types` 缺省时按 `full_attention_interval`（默认 4）自动生成：每隔
///   interval 层放一个 `full_attention`，其余为 `linear_attention`；
/// - 文本部分为 dense（py `base_model_ep_plan = None`，无 MoE），故 `num_experts = 0`。
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub family: ModelName,
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
    /// 部分旋转因子；`rotary_dim = head_dim × partial_rotary_factor`。
    pub partial_rotary_factor: f32,
    pub tie_word_embeddings: bool,
    pub layer_spec: LayerSpec,
    pub qkv_bias: bool,
    pub use_qk_norm: bool,
    pub rope_scaling: Option<HashMap<String, Value>>,
    pub eos_token_id: usize,
    pub eos_token_ids: Vec<usize>,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub intermediate_size: usize,
    // GatedDeltaNet（线性注意力）块维度；混合架构中 layer_types==linear_attention 的层使用。
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

impl TextConfig {
    pub fn from_hf(hf: &HfConfig) -> Self {
        let family = ModelName::parse(&hf.model_type);
        let num_attention_heads = hf.num_attention_heads.max(1);
        let head_dim = hf
            .head_dim
            .unwrap_or_else(|| hf.hidden_size / num_attention_heads);

        // py `__post_init__`: partial_rotary_factor 默认 0.25 → rotary_dim = head_dim × factor。
        let partial_rotary_factor = hf
            .partial_rotary_factor
            .unwrap_or(DEFAULT_PARTIAL_ROTARY_FACTOR);
        let rotary_dim = hf
            .rotary_dim
            .unwrap_or_else(|| ((head_dim as f32) * partial_rotary_factor) as usize);

        let num_key_value_heads = hf.num_key_value_heads.unwrap_or(num_attention_heads);
        let intermediate_size = hf.intermediate_size.unwrap_or(hf.hidden_size);
        let max_window_layers = hf.max_window_layers.unwrap_or(hf.num_hidden_layers);

        // qwen3_5 文本部分为 dense（无 MoE）；num_experts=0 时 LayerSpec::ffn 恒返回 Dense。
        let num_experts = hf.num_experts.unwrap_or(0);
        let num_experts_per_tok = hf.num_experts_per_tok.unwrap_or(0);
        let moe_intermediate_size = hf.moe_intermediate_size.unwrap_or(intermediate_size);
        let router_scoring = RouterScoringKind::from_hf(hf.scoring_func.as_deref(), family.clone());
        let use_routing_bias = hf.use_routing_bias.unwrap_or(false);
        let decoder_sparse_step = hf.decoder_sparse_step.max(1);
        // Qwen3 系列默认启用 qk-norm（与 qwen3_moe::Config 判定一致）。
        let use_qk_norm = hf.use_qk_norm
            || matches!(hf.model_type.as_str(), "qwen3_5" | "qwen3_5_text");

        // py `__post_init__`: layer_types 缺省时按 full_attention_interval 生成。
        // bool((i+1) % interval) 为真 → linear_attention，为假（整除）→ full_attention。
        let layer_types = hf.layer_types.clone().or_else(|| {
            let interval = hf
                .full_attention_interval
                .unwrap_or(DEFAULT_FULL_ATTENTION_INTERVAL)
                .max(1);
            Some(
                (0..hf.num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % interval == 0 {
                            "full_attention".to_string()
                        } else {
                            "linear_attention".to_string()
                        }
                    })
                    .collect(),
            )
        });

        let layer_spec = LayerSpec {
            num_hidden_layers: hf.num_hidden_layers,
            use_sliding_window: hf.use_sliding_window,
            max_window_layers,
            layer_types,
            mlp_only_layers: hf.mlp_only_layers.clone(),
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            intermediate_size,
            norm_topk_prob: hf.norm_topk_prob,
            decoder_sparse_step,
            router_scoring,
            use_routing_bias,
        };

        let eos_token_id = hf.eos_token_id;
        let eos_token_ids = vec![eos_token_id];

        Self {
            family,
            vocab_size: hf.vocab_size,
            hidden_size: hf.hidden_size,
            num_hidden_layers: hf.num_hidden_layers,
            num_attention_heads: hf.num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings: hf.max_position_embeddings,
            rms_norm_eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta.unwrap_or(10000),
            rotary_dim,
            partial_rotary_factor,
            tie_word_embeddings: hf.tie_word_embeddings,
            layer_spec,
            qkv_bias: hf.qkv_bias,
            use_qk_norm,
            rope_scaling: hf.rope_scaling.clone(),
            eos_token_id,
            eos_token_ids,
            max_window_layers,
            use_sliding_window: hf.use_sliding_window,
            sliding_window: hf.sliding_window,
            intermediate_size,
            linear_num_key_heads: hf
                .linear_num_key_heads
                .unwrap_or(DEFAULT_LINEAR_NUM_KEY_HEADS),
            linear_num_value_heads: hf
                .linear_num_value_heads
                .unwrap_or(DEFAULT_LINEAR_NUM_VALUE_HEADS),
            linear_key_head_dim: hf.linear_key_head_dim.unwrap_or(DEFAULT_LINEAR_KEY_HEAD_DIM),
            linear_value_head_dim: hf.linear_value_head_dim.unwrap_or(DEFAULT_LINEAR_VALUE_HEAD_DIM),
            linear_conv_kernel_dim: hf
                .linear_conv_kernel_dim
                .unwrap_or(DEFAULT_LINEAR_CONV_KERNEL_DIM),
        }
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
/// `VisionConfig::default()`。
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
        let text_config = TextConfig::from_hf(text_hf);
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
            vision_end_token_id: hf.vision_end_token_id.unwrap_or(DEFAULT_VISION_END_TOKEN_ID),
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
    use crate::model_spec::AttentionKind;

    #[test]
    fn text_config_derives_rotary_dim_and_layer_types() {
        // 嵌套 config：head_dim=256、partial_rotary_factor 缺省(0.25)、num_hidden_layers=8。
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

        // layer_types 自动生成（full_attention_interval=4）：索引 3、7 为 full，其余 linear。
        assert_eq!(text.layer_spec.len(), 8);
        assert_eq!(text.layer_spec.attention(0), AttentionKind::Linear);
        assert_eq!(text.layer_spec.attention(2), AttentionKind::Linear);
        assert_eq!(text.layer_spec.attention(3), AttentionKind::Full);
        assert_eq!(text.layer_spec.attention(4), AttentionKind::Linear);
        assert_eq!(text.layer_spec.attention(7), AttentionKind::Full);

        // dense 文本部分：无专家。
        assert!(matches!(
            text.layer_spec.ffn(0),
            crate::model_spec::FfnKind::Dense { .. }
        ));
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
