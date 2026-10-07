//! `AutoConfig`：对齐 HuggingFace `configuration_auto.AutoConfig.from_pretrained`
//! 的惯用门面。Python 版依赖 importlib + 运行时类对象 + 懒加载注册表
//! （`CONFIG_MAPPING`）按 `model_type` 动态分发到具体配置类；Rust 无对应机制，
//! 这里复用 eLLM 既有的静态分发链（`HfConfig` → `ModelName::parse` →
//! `profile_for` → `TextConfig::from_hf`，由 `load_text_config` 封装），仅补齐
//! “给定模型目录、自动拼标准文件名”的门面语义，消除各消费方重复的路径样板。

use std::path::Path;

use crate::config::GenerationConfig;
use crate::model_family::load_text_config;
use crate::model_spec::TextConfig;

/// 一次 `from_pretrained` 解析出的配置集合。
///
/// - `text`：模型结构蓝图（`config.json`），等价于 HF 的 `AutoConfig` 主体。
/// - `generation`：采样默认值（`generation_config.json`），文件缺失时为 `None`，
///   与既有消费方 `.ok()` 的容错行为一致。
#[derive(Debug, Clone)]
pub struct AutoConfig {
    pub text: TextConfig,
    pub generation: Option<GenerationConfig>,
}

impl AutoConfig {
    /// 从模型目录加载配置。目录下需存在 `config.json`；
    /// `generation_config.json` 可选。
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> anyhow::Result<Self> {
        let dir = model_dir.as_ref();
        let text = load_text_config(dir.join("config.json"))
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        let generation = GenerationConfig::load_from_file(dir.join("generation_config.json")).ok();
        Ok(Self { text, generation })
    }
}
