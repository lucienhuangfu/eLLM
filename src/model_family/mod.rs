use std::path::Path;

use crate::config::HfConfig;
use crate::model_spec::{FamilyProfile, ModelName, TextConfig};

pub mod qwen3_5;
pub mod qwen3_moe;

/// 按 family 分发 profile（父模块引用子模块，无反向依赖）。未特化的 family 用
/// `FamilyProfile::default()`（中性默认：rotary_dim=head_dim、无线性注意力、
/// qk_norm/routing_bias 关闭）。
pub fn profile_for(family: &ModelName, model_type: &str) -> FamilyProfile {
    match family {
        ModelName::Qwen3_5 => qwen3_5::profile(),
        ModelName::Qwen => qwen3_moe::profile(model_type),
        ModelName::MiniMaxM2 => FamilyProfile {
            routing_bias: true,
            ..Default::default()
        },
        _ => FamilyProfile::default(),
    }
}

/// 通用文本配置加载入口：读 config.json → 解析 family → 注入对应 profile →
/// `TextConfig::from_hf`。纯文本模型（bin / runtime）走此；多模态模型走
/// `qwen3_5::Config::load_from_file`（额外解析 vision 子配置与视觉特殊 token）。
pub fn load_text_config<P: AsRef<Path>>(path: P) -> Result<TextConfig, Box<dyn std::error::Error>> {
    let hf = HfConfig::load_from_file(path)?;
    let family = ModelName::parse(&hf.model_type);
    Ok(TextConfig::from_hf(
        &hf,
        &profile_for(&family, &hf.model_type),
    ))
}
