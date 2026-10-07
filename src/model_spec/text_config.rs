//! Family-agnostic whole-model **text** config: the single `TextConfig` shared
//! by every model family, plus the `FamilyProfile` value object that injects
//! per-family defaults into `TextConfig::from_hf`.
//!
//! 家族无关的整模型**文本**配置：所有 family 共用的唯一 `TextConfig`，以及向
//! `TextConfig::from_hf` 注入各 family 默认值的 `FamilyProfile` 值对象。
//!
//! 分层：`config::HfConfig`（原始 JSON）→ `TextConfig::from_hf(hf, profile)`
//! （应用 family 默认值 + 派生）。`FamilyProfile` 只描述默认值、不含分发逻辑；
//! 各 family 目录（`model_family::qwen3_moe` / `qwen3_5`）提供自己的 profile，
//! `model_family::profile_for` 按 `ModelName` 分发。如此 `model_spec` 不反向依赖
//! 任何具体 family。

use std::collections::HashMap;

use serde_json::Value;

use crate::config::HfConfig;

use super::layer_spec::LayerSpec;
use super::model_name::ModelName;
use super::router_scoring::RouterScoringKind;

/// 各 family 的 GatedDeltaNet（线性注意力）块默认维度；非混合模型全 0
/// （其 layer 永不解析为 `AttentionKind::Linear`）。
#[derive(Debug, Clone)]
pub struct LinearDefaults {
    pub num_k: usize,
    pub num_v: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
}

impl Default for LinearDefaults {
    fn default() -> Self {
        Self {
            num_k: 0,
            num_v: 0,
            key_dim: 0,
            value_dim: 0,
            conv_kernel: 0,
        }
    }
}

/// 各 family 的默认值集合，注入 `TextConfig::from_hf` 以消除 per-family 分支。
/// `Default` 为中性值（partial_rotary=1.0 即 rotary_dim=head_dim、无 interval、
/// linear 全 0、qk_norm/routing_bias 关闭），未特化的 family 直接用之。
#[derive(Debug, Clone)]
pub struct FamilyProfile {
    /// `rotary_dim = head_dim × partial_rotary_factor`；qwen3_5 为 0.25，其余 1.0。
    pub partial_rotary_factor: f32,
    /// `layer_types` 缺省时按此间隔生成（每隔 interval 层一个 full_attention）；
    /// `None` 表示不自动生成（沿用 `hf.layer_types`，可能为 None）。
    pub full_attention_interval: Option<usize>,
    /// 线性注意力块默认维度。
    pub linear_defaults: LinearDefaults,
    /// 是否默认启用 qk-norm（qwen3 系为 true，qwen2 为 false）。
    pub qk_norm: bool,
    /// MoE 路由是否默认带 bias（MiniMaxM2 为 true）。
    pub routing_bias: bool,
}

impl Default for FamilyProfile {
    fn default() -> Self {
        Self {
            partial_rotary_factor: 1.0,
            full_attention_interval: None,
            linear_defaults: LinearDefaults::default(),
            qk_norm: false,
            routing_bias: false,
        }
    }
}

/// 整模型文本配置（family-agnostic）。字段集为原 `qwen3_moe::Config` 全集加
/// `partial_rotary_factor`（来自 qwen3_5）。所有 family 的文本主干共用此结构，
/// family 差异通过 `FamilyProfile` 注入 `from_hf`。
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
    // GatedDeltaNet（线性注意力）块维度；非混合模型为 0。
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

impl TextConfig {
    /// 从原始 `HfConfig` 派生文本配置。family 差异全部经 `profile` 注入，
    /// 故本函数是所有 family 唯一的 `from_hf` 实现。
    pub fn from_hf(hf: &HfConfig, profile: &FamilyProfile) -> Self {
        let family = ModelName::parse(&hf.model_type);
        let head_dim = hf
            .head_dim
            .unwrap_or_else(|| hf.hidden_size / hf.num_attention_heads.max(1));
        let num_key_value_heads = hf
            .num_key_value_heads
            .unwrap_or(hf.num_attention_heads.max(1));

        // rotary_dim = head_dim × partial_rotary_factor（qwen3_5 默认 0.25 → 64；
        // 其余 family partial_rotary_factor=1.0 → rotary_dim=head_dim）。
        let partial_rotary_factor = hf
            .partial_rotary_factor
            .unwrap_or(profile.partial_rotary_factor);
        let rotary_dim = hf
            .rotary_dim
            .unwrap_or_else(|| ((head_dim as f32) * partial_rotary_factor) as usize);

        let intermediate_size = hf
            .intermediate_size
            .unwrap_or_else(|| hf.moe_intermediate_size.unwrap_or(hf.hidden_size));
        let moe_intermediate_size = hf.moe_intermediate_size.unwrap_or(intermediate_size);
        let num_experts = hf.num_experts.unwrap_or(0);
        let num_experts_per_tok = hf.num_experts_per_tok.unwrap_or(0);
        let max_window_layers = hf.max_window_layers.unwrap_or(hf.num_hidden_layers);
        let router_scoring = RouterScoringKind::from_hf(hf.scoring_func.as_deref(), family.clone());
        let use_routing_bias = hf.use_routing_bias.unwrap_or(profile.routing_bias);
        let decoder_sparse_step = hf.decoder_sparse_step.max(1);
        let use_qk_norm = hf.use_qk_norm || profile.qk_norm;

        // layer_types：显式给出则用之；否则若 profile 指定 full_attention_interval，
        // 按 (i+1)%interval==0 → full_attention，其余 linear_attention 生成。
        let layer_types = hf.layer_types.clone().or_else(|| {
            profile.full_attention_interval.map(|interval| {
                let interval = interval.max(1);
                (0..hf.num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % interval == 0 {
                            "full_attention".to_string()
                        } else {
                            "linear_attention".to_string()
                        }
                    })
                    .collect()
            })
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
                .unwrap_or(profile.linear_defaults.num_k),
            linear_num_value_heads: hf
                .linear_num_value_heads
                .unwrap_or(profile.linear_defaults.num_v),
            linear_key_head_dim: hf
                .linear_key_head_dim
                .unwrap_or(profile.linear_defaults.key_dim),
            linear_value_head_dim: hf
                .linear_value_head_dim
                .unwrap_or(profile.linear_defaults.value_dim),
            linear_conv_kernel_dim: hf
                .linear_conv_kernel_dim
                .unwrap_or(profile.linear_defaults.conv_kernel),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_spec::{AttentionKind, FfnKind};

    /// qwen3_5 的 profile（与 `model_family::qwen3_5::profile()` 一致）。
    fn qwen3_5_profile() -> FamilyProfile {
        FamilyProfile {
            partial_rotary_factor: 0.25,
            full_attention_interval: Some(4),
            linear_defaults: LinearDefaults {
                num_k: 16,
                num_v: 32,
                key_dim: 128,
                value_dim: 128,
                conv_kernel: 4,
            },
            qk_norm: true,
            routing_bias: false,
        }
    }

    #[test]
    fn qwen3_5_derives_rotary_dim_and_layer_types() {
        // head_dim=256、partial_rotary_factor 缺省(profile 0.25)、num_hidden_layers=8。
        let json = r#"{
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 8,
            "hidden_size": 4096,
            "num_attention_heads": 16,
            "head_dim": 256
        }"#;
        let hf: HfConfig = serde_json::from_str(json).unwrap();
        let text = TextConfig::from_hf(&hf, &qwen3_5_profile());

        assert!(matches!(text.family, ModelName::Qwen3_5));
        // rotary_dim = 256 × 0.25 = 64。
        assert_eq!(text.rotary_dim, 64);
        assert_eq!(text.head_dim, 256);
        assert!(text.use_qk_norm);

        // linear 默认来自 profile。
        assert_eq!(text.linear_num_key_heads, 16);
        assert_eq!(text.linear_num_value_heads, 32);
        assert_eq!(text.linear_key_head_dim, 128);
        assert_eq!(text.linear_value_head_dim, 128);
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
    fn qwen3_moe_profile_keeps_rotary_dim_equal_head_dim() {
        // qwen3 profile：partial_rotary=1.0（默认），qk_norm=true，linear 全 0。
        let profile = FamilyProfile {
            qk_norm: true,
            ..Default::default()
        };
        let json = r#"{
            "model_type": "qwen3_moe",
            "num_hidden_layers": 4,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "head_dim": 128,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 512,
            "decoder_sparse_step": 1
        }"#;
        let hf: HfConfig = serde_json::from_str(json).unwrap();
        let text = TextConfig::from_hf(&hf, &profile);

        assert!(matches!(text.family, ModelName::Qwen));
        // rotary_dim = head_dim（factor 1.0）。
        assert_eq!(text.rotary_dim, 128);
        assert!(text.use_qk_norm);
        // MoE layer_spec。
        assert!(matches!(text.layer_spec.ffn(0), FfnKind::SparseMoe { .. }));
        // 无 interval → layer_types=None → attention 默认 Full。
        assert_eq!(text.layer_spec.attention(0), AttentionKind::Full);
        // linear 默认全 0。
        assert_eq!(text.linear_num_key_heads, 0);
        assert_eq!(text.linear_conv_kernel_dim, 0);
    }

    #[test]
    fn qwen2_profile_disables_qk_norm() {
        // qwen2：profile 全默认（qk_norm=false）。
        let profile = FamilyProfile::default();
        let json = r#"{
            "model_type": "qwen2",
            "num_hidden_layers": 2,
            "hidden_size": 1024,
            "num_attention_heads": 8
        }"#;
        let hf: HfConfig = serde_json::from_str(json).unwrap();
        let text = TextConfig::from_hf(&hf, &profile);

        assert!(!text.use_qk_norm);
        // head_dim = 1024 / 8 = 128；rotary_dim = 128 × 1.0。
        assert_eq!(text.head_dim, 128);
        assert_eq!(text.rotary_dim, 128);
    }
}
