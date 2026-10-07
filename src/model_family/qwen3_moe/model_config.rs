//! qwen 系（qwen2 / qwen3 / qwen3_moe）的 family profile 与配置入口。
//!
//! 文本主干配置已通用化为 `model_spec::TextConfig`；本模块只承载 qwen 系的
//! `FamilyProfile`（默认值）与 `Config` 入口别名，维持“每模型一目录”的组织约定。

use crate::model_spec::FamilyProfile;

/// qwen 系文本配置入口：通用 `TextConfig` 的 family 别名。
pub use crate::model_spec::TextConfig as Config;

/// qwen 系（`ModelName::Qwen`）的 family 默认值。
///
/// `qk_norm` 由 `model_type` 区分 qwen2 / qwen3：`ModelName::Qwen` 同时涵盖
/// qwen2 / qwen2_moe / qwen3 / qwen3_moe，其中仅 qwen3 系默认启用 qk-norm
/// （与原 `qwen3_moe::Config::from_hf` 判定一致）。qwen 系为 dense 或标准 MoE，
/// 无 GatedDeltaNet 线性注意力，故 linear 默认全 0；rotary_dim = head_dim
/// （partial_rotary_factor=1.0）；不自动生成 layer_types。
pub fn profile(model_type: &str) -> FamilyProfile {
    FamilyProfile {
        qk_norm: matches!(model_type, "qwen3" | "qwen3_moe"),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::profile;

    #[test]
    fn qwen3_enables_qk_norm() {
        assert!(profile("qwen3").qk_norm);
        assert!(profile("qwen3_moe").qk_norm);
    }

    #[test]
    fn qwen2_disables_qk_norm() {
        assert!(!profile("qwen2").qk_norm);
        assert!(!profile("qwen2_moe").qk_norm);
    }

    #[test]
    fn qwen_profile_is_neutral_except_qk_norm() {
        let p = profile("qwen3_moe");
        assert_eq!(p.partial_rotary_factor, 1.0);
        assert!(p.full_attention_interval.is_none());
        assert_eq!(p.linear_defaults.num_k, 0);
        assert_eq!(p.linear_defaults.conv_kernel, 0);
        assert!(!p.routing_bias);
    }
}
