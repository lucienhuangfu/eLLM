//! `AutoFeatureExtractor`：对齐 HuggingFace `feature_extraction_auto.AutoFeatureExtractor`
//! 的 **config-only 占位门面**。
//!
//! 说明：qwen3_5 是纯视觉-语言模型，**没有音频模态**——`feature_extraction_auto.py` 的
//! `FEATURE_EXTRACTOR_MAPPING_NAMES` 里也不含 qwen3_5（只有 whisper/wav2vec2/encodec 等
//! 音频 family）。因此本模块不实现任何音频特征提取数学（mel 频谱 / STFT / 归一化），
//! 仅提供与 [`super::image_processor`] / [`super::video_processor`] 一致的
//! `from_pretrained` 门面：定位并解析 `preprocessor_config.json`（或嵌套
//! `processor_config.json` 的 `feature_extractor` 子对象）为强类型配置，供未来接入
//! 音频 family 时使用。
//!
//! Python 侧 `AutoFeatureExtractor.from_pretrained` 按 `feature_extractor_type` /
//! `model_type` 经 `FEATURE_EXTRACTOR_MAPPING` 动态分发到具体音频处理器；Rust 无
//! importlib，且 eLLM 尚无音频处理器实现，故此处止步于配置解析层。

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 音频特征提取器同样使用 `preprocessor_config.json`（HF `FEATURE_EXTRACTOR_NAME`）。
const FEATURE_EXTRACTOR_NAME: &str = "preprocessor_config.json";
/// HF `PROCESSOR_NAME`（嵌套 processor 配置）。
const PROCESSOR_NAME: &str = "processor_config.json";

/// 音频特征提取器配置，对应 `preprocessor_config.json` 的常见字段（Whisper/Wav2Vec2 等）。
/// 全部字段可缺省；未识别字段收进 `extra`。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureExtractorConfig {
    #[serde(default)]
    pub feature_extractor_type: Option<String>,
    #[serde(default)]
    pub feature_size: Option<usize>,
    #[serde(default)]
    pub sampling_rate: Option<usize>,
    #[serde(default)]
    pub num_mel_bins: Option<usize>,
    #[serde(default)]
    pub n_fft: Option<usize>,
    #[serde(default)]
    pub hop_length: Option<usize>,
    #[serde(default)]
    pub win_length: Option<usize>,
    #[serde(default)]
    pub n_samples: Option<usize>,
    #[serde(default)]
    pub nb_max_frames: Option<usize>,
    #[serde(default)]
    pub chunk_length: Option<usize>,
    #[serde(default)]
    pub padding_value: Option<f32>,
    #[serde(default)]
    pub do_normalize: Option<bool>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// 音频特征提取器 config-only 门面。
#[derive(Debug, Clone)]
pub struct AutoFeatureExtractor {
    pub config: FeatureExtractorConfig,
}

impl AutoFeatureExtractor {
    /// 从模型目录加载：优先嵌套 `processor_config.json` 的 `feature_extractor` 子对象，
    /// 其次 `preprocessor_config.json`（对齐 HF `get_feature_extractor_config` 优先级）。
    ///
    /// 注意：qwen3_5 目录不含音频配置，此调用会因找不到文件而报错——这是预期行为，
    /// 因为该 family 无音频模态。
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> anyhow::Result<Self> {
        let dir = model_dir.as_ref();
        let config = load_feature_extractor_config(dir)?;
        Ok(Self { config })
    }

    /// 由显式配置构造。
    pub fn from_config(config: FeatureExtractorConfig) -> Self {
        Self { config }
    }
}

/// 读取并解析音频特征提取器配置 JSON。
fn load_feature_extractor_config(dir: &Path) -> anyhow::Result<FeatureExtractorConfig> {
    let processor_path = dir.join(PROCESSOR_NAME);
    if processor_path.exists() {
        let raw = std::fs::read_to_string(&processor_path)
            .map_err(|e| anyhow::anyhow!("unable to read {}: {e}", processor_path.display()))?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("unable to parse {}: {e}", processor_path.display()))?;
        if let Some(sub) = value.get("feature_extractor") {
            return serde_json::from_value(sub.clone()).map_err(|e| {
                anyhow::anyhow!(
                    "unable to deserialize feature_extractor in {}: {e}",
                    processor_path.display()
                )
            });
        }
    }

    let fe_path = dir.join(FEATURE_EXTRACTOR_NAME);
    if fe_path.exists() {
        let raw = std::fs::read_to_string(&fe_path)
            .map_err(|e| anyhow::anyhow!("unable to read {}: {e}", fe_path.display()))?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("unable to parse {}: {e}", fe_path.display()))?;
        // 音频配置以 feature_extractor_type / feature_size 等字段为标志；
        // 若该文件其实是图像配置（含 image_processor_type 而无音频字段），视为不适用。
        if value.get("feature_extractor_type").is_some()
            || value.get("feature_size").is_some()
            || value.get("num_mel_bins").is_some()
        {
            return serde_json::from_value(value).map_err(|e| {
                anyhow::anyhow!(
                    "unable to deserialize feature extractor {}: {e}",
                    fe_path.display()
                )
            });
        }
    }

    anyhow::bail!(
        "no audio feature extractor config found in {} (expected {PROCESSOR_NAME} with \
         `feature_extractor`, or {FEATURE_EXTRACTOR_NAME} with audio fields); note that \
         vision-only families such as qwen3_5 have no audio modality",
        dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_whisper_style_config() {
        let json = r#"{
            "feature_extractor_type": "WhisperFeatureExtractor",
            "feature_size": 80,
            "num_mel_bins": 80,
            "sampling_rate": 16000,
            "hop_length": 160,
            "n_fft": 400,
            "padding_value": 0.0,
            "unknown_field": 1
        }"#;
        let cfg: FeatureExtractorConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.feature_extractor_type.as_deref(),
            Some("WhisperFeatureExtractor")
        );
        assert_eq!(cfg.feature_size, Some(80));
        assert_eq!(cfg.sampling_rate, Some(16000));
        assert!(cfg.extra.contains_key("unknown_field"));
    }

    #[test]
    fn from_config_roundtrip() {
        let cfg = FeatureExtractorConfig {
            feature_size: Some(64),
            ..Default::default()
        };
        let fe = AutoFeatureExtractor::from_config(cfg);
        assert_eq!(fe.config.feature_size, Some(64));
    }
}
