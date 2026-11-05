//! `TextConfig`：Hugging Face `config.json` 的**纯数据镜像**，本仓库唯一的模型配置结构体。
//!
//! 设计原则（两平面）
//!
//! - **数据平面（本文件）**：字段名 = JSON 键名，结构级 `#[serde(default)]`，缺键取
//!   哨兵值（`0` / `false` / `""` / 空 `Vec`）。本文件除 `apply_family_defaults` 一处
//!   按 `model_type` 补齐**数值缺省**外，不含任何形态判断。
//! - **判断平面（`src/transformer/**`）**：逐层形态（attention linear/full、FFN
//!   dense/MoE、router softmax/sigmoid）与架构事实（qk-norm、zero-centered
//!   RMSNorm）都由真正使用它们的那个层文件自己判定，对应各家
//!   `modeling_<name>.py` 的类体与 `__init__` 构造点。
//!
//! 三段管线：`serde 直读 → apply_family_defaults（B 类数据缺省）→ derive（无分支算术派生）
//! → validate（加载期硬校验）`。对应 Python 的 dataclass 默认值 + `__post_init__`。
//!
//! # `Option` 的红线
//!
//! `#[serde(default)]` 只覆盖「键缺失」，**不覆盖「值是 null」**。已核 `checkpoints/`
//! 下全部 `config.json`，显式取 `null` 的键只有三个：`rope_scaling`、`pad_token_id`、
//! `sliding_window` —— 它们必须是 `Option`，否则反序列化直接失败。
//!
//! # 未被消费的 JSON 键（G5 盘点）
//!
//! 以下键存在于真实 ckpt，本结构体**有意不收**，逐项定性：
//!
//! - `mtp_num_hidden_layers` / `mtp_use_dedicated_embeddings`（Qwen3.8）：MTP 投机解码，
//!   待接。Python dense 实现以 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` 忽略。
//! - `mamba_ssm_dtype`（Qwen3.8）：GatedDeltaNet 内部 dtype，本引擎按 `T` 决定。不需要。
//! - `attention_dropout` / `initializer_range` / `num_logits_to_keep`：训练期参数。不需要。
//! - `qk_norm_type`（MiniMax，值 `"per_layer"`）：**待接** —— 现层代码只做 per-head。
//! - `shared_intermediate_size`（MiniMax，值 `0`）：**待接** —— 共享专家未实现；`0`
//!   恰好表示无共享专家，故当前不消费它是安全的。
//! - `sliding_window` / `use_sliding_window`：滑窗注意力已从本引擎移除，仅保留
//!   `sliding_window` 字段以接受显式 `null`。
//! - `image_token_id` / `video_token_id` / `vision_start_token_id` / `vision_end_token_id`
//!   与 `vision_config`：多模态装配，见 [`crate::config::RootConfig`]。

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use super::vision_config::{
    VisionConfig, VisionTokens, DEFAULT_IMAGE_TOKEN_ID, DEFAULT_VIDEO_TOKEN_ID,
    DEFAULT_VISION_END_TOKEN_ID, DEFAULT_VISION_START_TOKEN_ID,
};

/// 整模型文本配置：`config.json`（扁平模型取顶层，多模态取 `text_config` 子对象）的
/// 1:1 数据镜像。字段的最终值由 [`TextConfig::load`] 保证完备 —— 过了 `load` 的配置
/// 永远持有已派生、已归一化、已通过校验的值，层代码可直接读，无需再兜底。
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct TextConfig {
    // ── 主干 ──
    /// 原始 `model_type` 字符串。层代码的形态/架构判定与 `apply_family_defaults`
    /// 都直接读它，不做枚举归一化。
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// `0` 表示未给出，由 `derive()` 取 `hidden_size / num_attention_heads`。
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: usize,
    /// `0` 表示未给出，由 `derive()` 取 `head_dim × partial_rotary_factor`。
    pub rotary_dim: usize,
    pub tie_word_embeddings: bool,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    /// `derive()` 生成，等价 Python 的 `eos_token_id` 单值列表。
    pub eos_token_ids: Vec<usize>,

    // ── attention 形态（原始字符串，判定在层代码）──
    /// 逐层类型清单。`derive()` 保证非空且长度 == `num_hidden_layers`，
    /// 已 `trim + lowercase`，取值含 `full_attention` / `linear_attention`。
    pub layer_types: Vec<String>,
    /// `layer_types` 缺省时的生成周期：每 `interval` 层的最后一层为 `full_attention`。
    /// 中性默认取 1（恒生成全 `full_attention`），qwen3_5 系取 4。
    pub full_attention_interval: usize,
    /// `rotary_dim = head_dim × partial_rotary_factor`。中性默认 1.0，qwen3_5 系 0.25。
    pub partial_rotary_factor: f32,
    /// G1：HF 各家族都把这个键写成注意力偏置的本名。早期实现按投影合称去读，
    /// 那个键在任何真实 ckpt 里都不存在，因此不保留旧拼写的别名。
    pub attention_bias: bool,
    /// Qwen3.8 的输出门控。HF `modeling_qwen3_5.py` L808 的门控形态为
    /// `attn_output * torch.sigmoid(gate)`（注：尽管 `output_gate_type` 配为
    /// `"swish"`，本 ckpt 的参考 forward 对该门仍直接用 sigmoid，未走 ACT2FN）。
    /// 门控已融合进 attention 算子的 col 循环收尾 epilogue；`attn_output_gate` 决定
    /// 是否有门，判定与实现在 `transformer/attention/full_attention.rs`。
    pub attn_output_gate: bool,
    pub output_gate_type: String,

    // ── GatedDeltaNet（线性注意力）块维度；仅混合模型需要，自检在线性注意力层 ──
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,

    // ── MoE 形态（判定在层代码）──
    /// MiniMax-M2.5 用 `num_local_experts` 表达同一含义，收为别名；`0` = 无 MoE。
    #[serde(alias = "num_local_experts")]
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// `0` 表示未给出，由 `derive()` 回落 `intermediate_size`。
    pub moe_intermediate_size: usize,
    /// 稀疏周期，`derive()` 保证 `>= 1`。
    pub decoder_sparse_step: usize,
    /// 强制 dense 的层号清单。
    pub mlp_only_layers: Vec<usize>,
    /// router 打分函数原始字符串，`validate()` 归一化并限定词表。
    pub scoring_func: String,
    pub use_routing_bias: bool,

    // ── 只能留 Option：真实 ckpt 里会显式写 null ──
    /// RoPE 参数表。HF 5.x 起该对象名为 `rope_parameters`（旧名 `rope_scaling`，
    /// 收为别名）；Qwen3.8 的 `rope_theta` 就藏在这张表里，`derive()` 会取出。
    #[serde(alias = "rope_parameters")]
    pub rope_scaling: Option<HashMap<String, Value>>,
    pub pad_token_id: Option<usize>,
    pub sliding_window: Option<usize>,
}

impl TextConfig {
    /// 读 `config.json` 并完成 `apply_family_defaults → derive → validate` 的完整管线。
    ///
    /// 多模态 ckpt（顶层含 `text_config` 子对象）以嵌套子配置为准，与 Python
    /// `Qwen3_5Config` 优先取 `text_config` 的行为一致。
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        Self::load_from_reader(BufReader::new(file))
    }

    pub fn load_from_reader<R: Read>(reader: R) -> Result<Self, Box<dyn std::error::Error>> {
        let root: super::RootConfig = serde_json::from_reader(reader)?;
        Ok(root.into_text_config())
    }

    /// 完整管线：数据缺省 → 算术派生 → 归一化与校验。`load` 与各 `RootConfig::into_*`
    /// 都只经此一处进入三段私有方法。
    pub fn finish(&mut self) {
        self.apply_family_defaults();
        self.derive();
        self.validate();
    }

    /// [`TextConfig::finish`] 的消费型版本。
    pub fn finished(mut self) -> Self {
        self.finish();
        self
    }

    /// B 类数据缺省补齐：**全仓库唯一**一处按 `model_type` 分支的配置逻辑，只补数值、
    /// 不选形态。对应各家 `configuration_<name>.py` 的 dataclass 类默认值与
    /// `__post_init__` 的 `setdefault`。
    fn apply_family_defaults(&mut self) {
        // qwen3_5 系（含嵌套子配置的 `qwen3_5_text`）：
        // py `__post_init__` 的 setdefault("partial_rotary_factor", 0.25) 与
        // kwargs.pop("full_attention_interval", 4)，以及 GatedDeltaNet 五个块维度默认。
        if matches!(self.model_type.as_str(), "qwen3_5" | "qwen3_5_text") {
            if self.partial_rotary_factor == 0.0 {
                self.partial_rotary_factor = 0.25;
            }
            if self.full_attention_interval == 0 {
                self.full_attention_interval = 4;
            }
            if self.linear_conv_kernel_dim == 0 {
                self.linear_conv_kernel_dim = 4;
            }
            if self.linear_key_head_dim == 0 {
                self.linear_key_head_dim = 128;
            }
            if self.linear_value_head_dim == 0 {
                self.linear_value_head_dim = 128;
            }
            if self.linear_num_key_heads == 0 {
                self.linear_num_key_heads = 16;
            }
            if self.linear_num_value_heads == 0 {
                self.linear_num_value_heads = 32;
            }
        }
        // 其余家族：中性默认。interval 取 1 使 `(i+1) % 1 == 0` 恒成立，
        // 生成的清单全为 full_attention —— 与「无 layer_types 即全因果注意力」等价。
        if self.partial_rotary_factor == 0.0 {
            self.partial_rotary_factor = 1.0;
        }
        if self.full_attention_interval == 0 {
            self.full_attention_interval = 1;
        }
    }

    /// 无分支的算术派生：`0` / 空清单就是「未给出」的哨兵。
    fn derive(&mut self) {
        if self.num_attention_heads == 0 {
            self.num_attention_heads = 1;
        }
        if self.head_dim == 0 {
            self.head_dim = self.hidden_size / self.num_attention_heads;
        }
        if self.num_key_value_heads == 0 {
            self.num_key_value_heads = self.num_attention_heads;
        }
        if self.rotary_dim == 0 {
            self.rotary_dim = (self.head_dim as f32 * self.partial_rotary_factor) as usize;
        }
        if self.intermediate_size == 0 {
            self.intermediate_size = self.moe_intermediate_size.max(self.hidden_size);
        }
        if self.moe_intermediate_size == 0 {
            self.moe_intermediate_size = self.intermediate_size;
        }
        if self.decoder_sparse_step == 0 {
            self.decoder_sparse_step = 1;
        }
        // rope_theta：顶层缺失时从 rope 参数表里取（Qwen3.8 把它写在 rope_parameters 内）。
        if self.rope_theta == 0 {
            self.rope_theta = self
                .rope_scaling
                .as_ref()
                .and_then(|params| params.get("rope_theta"))
                .and_then(Value::as_u64)
                .unwrap_or(10_000) as usize;
        }
        // layer_types 缺省按 interval 生成。
        if self.layer_types.is_empty() {
            self.layer_types = (0..self.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % self.full_attention_interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect();
        }
        // router 打分的通用缺省：无家族兜底（MiniMax 的 ckpt 本就显式写了 sigmoid）。
        if self.scoring_func.is_empty() {
            self.scoring_func = "softmax".to_string();
        }
        self.eos_token_ids = vec![self.eos_token_id];
    }

    /// 归一化 + 加载期硬校验：把今天的静默兜底换成显式失败。
    fn validate(&mut self) {
        assert!(
            self.hidden_size > 0 && self.num_hidden_layers > 0,
            "config.json 缺少必填字段: {self:?}"
        );
        assert!(
            self.full_attention_interval > 0,
            "full_attention_interval 不得为 0"
        );
        for t in &mut self.layer_types {
            *t = t.trim().to_ascii_lowercase();
        }
        assert_eq!(
            self.layer_types.len(),
            self.num_hidden_layers,
            "layer_types 长度必须等于 num_hidden_layers"
        );
        for t in &self.layer_types {
            assert!(
                t.contains("linear") || t.contains("full"),
                "未知 layer_type: {t}"
            );
        }
        self.scoring_func = self.scoring_func.trim().to_ascii_lowercase();
        assert!(
            self.scoring_func.contains("softmax") || self.scoring_func.contains("sigmoid"),
            "未知 scoring_func: {}",
            self.scoring_func
        );
    }

    // ── 只读便捷访问（不含形态判断；判断在层代码）──

    /// 第 `layer_idx` 层的原始 layer_type 字符串。`validate()` 已保证清单完备，
    /// 越界只在手工构造的 `TextConfig` 上才可能发生。
    pub fn layer_type(&self, layer_idx: usize) -> Option<&str> {
        self.layer_types.get(layer_idx).map(|s| s.as_str())
    }
}

/// 顶层 `config.json`：多模态模型把文本超参嵌在 `text_config` 里，并额外提供
/// `vision_config` 与四个视觉特殊 token id；扁平模型（qwen3 / qwen3_moe /
/// minimax_m2）只有展平字段，子配置皆为 `None`。
///
/// 对应 Python 的 `Qwen3_5Config`（`sub_configs = {"text_config", "vision_config"}`）。
/// 取代旧 `qwen3_5::Config::from_hf` 里 `text_config.as_deref().unwrap_or(&hf)` 的
/// 手工回落 —— 这里由 [`RootConfig::into_text_config`] 单点表达「嵌套优先」。
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct RootConfig {
    /// 扁平 ckpt 的文本字段直接落在这里。
    #[serde(flatten)]
    pub text: TextConfig,
    /// 嵌套 ckpt（Qwen3.8）的文本子配置；其 `model_type` 为 `qwen3_5_text`。
    pub text_config: Option<Box<TextConfig>>,
    pub vision_config: Option<VisionConfig>,
    pub image_token_id: Option<usize>,
    pub video_token_id: Option<usize>,
    pub vision_start_token_id: Option<usize>,
    pub vision_end_token_id: Option<usize>,
}

impl RootConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        Ok(serde_json::from_reader(BufReader::new(file))?)
    }

    /// 嵌套 `text_config` 优先，缺失时用顶层展平字段，然后跑完整管线。
    pub fn into_text_config(self) -> TextConfig {
        let mut text = self.text_config.map(|boxed| *boxed).unwrap_or(self.text);
        text.finish();
        text
    }

    /// 一次 `config.json` 解析出的多模态装配：文本配置 + 视觉配置 + 视觉特殊 token。
    pub fn into_multimodal(self) -> (TextConfig, VisionConfig, VisionTokens) {
        let vision = self.vision_config.unwrap_or_default();
        let tokens = VisionTokens {
            image_token_id: self.image_token_id.unwrap_or(DEFAULT_IMAGE_TOKEN_ID),
            video_token_id: self.video_token_id.unwrap_or(DEFAULT_VIDEO_TOKEN_ID),
            vision_start_token_id: self
                .vision_start_token_id
                .unwrap_or(DEFAULT_VISION_START_TOKEN_ID),
            vision_end_token_id: self
                .vision_end_token_id
                .unwrap_or(DEFAULT_VISION_END_TOKEN_ID),
        };
        let text = self
            .text_config
            .map(|boxed| *boxed)
            .unwrap_or(self.text)
            .finished();
        (text, vision, tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_from(json: &str) -> TextConfig {
        TextConfig::load_from_reader(json.as_bytes()).unwrap()
    }

    #[test]
    fn flat_dense_config_derives_head_dim_and_all_full_layers() {
        // Qwen3-0.6B 形状：无 layer_types / interval / head_dim 之外的缺省。
        let c = config_from(
            r#"{
                "model_type": "qwen3",
                "vocab_size": 151936,
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 40960,
                "rms_norm_eps": 1e-06,
                "rope_theta": 1000000,
                "eos_token_id": 151645
            }"#,
        );
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.num_key_value_heads, 8);
        // partial_rotary_factor 中性默认 1.0 → rotary_dim == head_dim。
        assert_eq!(c.rotary_dim, 128);
        // interval 中性默认 1 → 28 层全 full_attention（等价旧的「无清单兜底 Full」）。
        assert_eq!(c.layer_types.len(), 28);
        assert_eq!(c.layer_types[0], "full_attention");
        assert!(c.layer_types.iter().all(|t| t == "full_attention"));
        // 无专家 → 层代码的三条件合取必然判 dense。
        assert_eq!(c.num_experts, 0);
        assert_eq!(c.decoder_sparse_step, 1);
        assert_eq!(c.scoring_func, "softmax");
        assert!(!c.use_routing_bias);
        assert_eq!(c.moe_intermediate_size, 3072);
        assert_eq!(c.eos_token_ids, vec![151645]);
    }

    #[test]
    fn qwen3_5_family_defaults_apply_only_to_that_model_type() {
        // 顶层多模态 + 嵌套 text_config：以嵌套为准，model_type 为 qwen3_5_text。
        let c = config_from(
            r#"{
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "num_hidden_layers": 8,
                    "hidden_size": 4096,
                    "num_attention_heads": 16,
                    "head_dim": 256,
                    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}
                }
            }"#,
        );
        assert_eq!(c.model_type, "qwen3_5_text");
        // qwen3_5 系：partial_rotary_factor 0.25 → rotary_dim = 256 × 0.25 = 64。
        assert_eq!(c.partial_rotary_factor, 0.25);
        assert_eq!(c.rotary_dim, 64);
        // GatedDeltaNet 块维度取该家族类默认。
        assert_eq!(c.linear_num_key_heads, 16);
        assert_eq!(c.linear_num_value_heads, 32);
        assert_eq!(c.linear_conv_kernel_dim, 4);
        // interval 4 → 索引 3、7 为 full，其余 linear。
        assert_eq!(c.layer_types.len(), 8);
        assert_eq!(c.layer_types[0], "linear_attention");
        assert_eq!(c.layer_types[2], "linear_attention");
        assert_eq!(c.layer_types[3], "full_attention");
        assert_eq!(c.layer_types[7], "full_attention");
        // rope_theta 藏在 rope_parameters 里，derive() 取出（顶层无该键）。
        assert_eq!(c.rope_theta, 10_000_000);
    }

    #[test]
    fn explicit_layer_types_win_over_interval_and_length_is_enforced() {
        let c = config_from(
            r#"{
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 4,
                "num_attention_heads": 24,
                "intermediate_size": 17408,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "FULL_Attention"]
            }"#,
        );
        // 显式清单不重生成，只归一化大小写。
        assert_eq!(c.layer_types[3], "full_attention");

        // 清单长度与层数不符 → 加载期硬失败（今天会静默兜底 Full）。
        let bad = r#"{
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "intermediate_size": 3072,
            "layer_types": ["full_attention", "full_attention"]
        }"#;
        let err = std::panic::catch_unwind(|| config_from(bad));
        assert!(err.is_err(), "layer_types 长度不符必须在 load 期 panic");
    }

    #[test]
    fn unknown_layer_type_and_scoring_func_are_rejected() {
        let bad_layer = r#"{
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "intermediate_size": 3072,
            "layer_types": ["sliding_attention"]
        }"#;
        assert!(std::panic::catch_unwind(|| config_from(bad_layer)).is_err());

        let bad_scoring = r#"{
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "intermediate_size": 3072,
            "scoring_func": "topk"
        }"#;
        assert!(std::panic::catch_unwind(|| config_from(bad_scoring)).is_err());
    }

    #[test]
    fn explicit_null_keys_still_deserialize() {
        // 红线：rope_scaling / pad_token_id / sliding_window 可显式为 null。
        let c = config_from(
            r#"{
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 1,
                "num_attention_heads": 16,
                "intermediate_size": 3072,
                "rope_scaling": null,
                "sliding_window": null,
                "use_sliding_window": false,
                "pad_token_id": null
            }"#,
        );
        assert!(c.rope_scaling.is_none());
        assert!(c.sliding_window.is_none());
        assert!(c.pad_token_id.is_none());
        // 顶层无 rope_theta → 通用缺省。
        assert_eq!(c.rope_theta, 10_000);
    }

    #[test]
    fn minimax_num_local_experts_alias_feeds_sparse_moe_inputs() {
        // G0：MiniMax 用 num_local_experts，且不写 decoder_sparse_step / mlp_only_layers。
        let c = config_from(
            r#"{
                "model_type": "minimax_m2",
                "hidden_size": 3072,
                "num_hidden_layers": 62,
                "num_attention_heads": 24,
                "intermediate_size": 1536,
                "num_local_experts": 256,
                "num_experts_per_tok": 8,
                "scoring_func": "sigmoid",
                "use_routing_bias": true
            }"#,
        );
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 8);
        // step 缺省 1 + mlp_only_layers 空 → 层代码判定 62 层全 SparseMoe。
        assert_eq!(c.decoder_sparse_step, 1);
        assert!(c.mlp_only_layers.is_empty());
        assert_eq!(c.scoring_func, "sigmoid");
        assert!(c.use_routing_bias);
        assert!(c.layer_types.iter().all(|t| t == "full_attention"));
    }

    #[test]
    fn attention_bias_reads_the_hf_key() {
        // G1：本仓库的键名就是 HF 真名 attention_bias。
        let c = config_from(
            r#"{"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 1,
                "num_attention_heads": 16, "intermediate_size": 3072, "attention_bias": true}"#,
        );
        assert!(c.attention_bias);
    }

    // ── 四份真实 ckpt 的快照：重构的主验证手段 ──
    //
    // P1/P2 声明为行为等价，故此处逐字段锁定 `(层数, layer_types 序列, 每层是否
    // sparse, head_dim, rotary_dim, num_experts, scoring_func)`。建图快照
    // （每层注意力是否 linear、是否分配 q_norm、是否 zero_centered）在
    // `transformer/attention/full_attention.rs` 的同名测试里。

    /// ckpt 缺失时跳过（返回 `None`）而非失败 —— 仓库不自带权重目录。
    fn load_ckpt(dir: &str) -> Option<TextConfig> {
        let path = format!("checkpoints/{dir}/config.json");
        if !std::path::Path::new(&path).exists() {
            eprintln!("skip snapshot: {path} not found");
            return None;
        }
        Some(TextConfig::load(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
    }

    /// 与 `FfnBlock::new` 的三条件合取同形的判定，仅用于快照断言。
    fn is_sparse(c: &TextConfig, layer_idx: usize) -> bool {
        !c.mlp_only_layers.contains(&layer_idx)
            && c.num_experts > 0
            && (layer_idx + 1) % c.decoder_sparse_step == 0
    }

    #[test]
    fn snapshot_qwen3_06b() {
        let Some(c) = load_ckpt("Qwen3-0.6B") else {
            return;
        };
        assert_eq!(c.model_type, "qwen3");
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.rotary_dim, 128); // partial 中性 1.0
        assert_eq!(c.rope_theta, 1_000_000);
        // 无 MoE 键 → 64 层全 dense。
        assert_eq!(c.num_experts, 0);
        assert_eq!(c.decoder_sparse_step, 1);
        assert_eq!(c.moe_intermediate_size, 3072); // 回落 intermediate_size
        for i in 0..c.num_hidden_layers {
            assert!(!is_sparse(&c, i), "layer {i} 应为 dense");
        }
        // 无 layer_types / interval → 中性 interval=1 → 全 full_attention。
        assert!(c.layer_types.iter().all(|t| t == "full_attention"));
        assert_eq!(c.scoring_func, "softmax");
        assert!(!c.use_routing_bias);
    }

    #[test]
    fn snapshot_qwen3_coder_30b_a3b() {
        let Some(c) = load_ckpt("Qwen3-Coder-30B-A3B-Instruct") else {
            return;
        };
        assert_eq!(c.model_type, "qwen3_moe");
        assert_eq!(c.num_hidden_layers, 48);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.num_key_value_heads, 4);
        assert_eq!(c.rotary_dim, 128);
        assert_eq!(c.rope_theta, 10_000_000);
        assert_eq!(c.num_experts, 128);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_intermediate_size, 768);
        assert_eq!(c.intermediate_size, 6144);
        // decoder_sparse_step = 1 → 全层 SparseMoe。
        for i in 0..c.num_hidden_layers {
            assert!(is_sparse(&c, i), "layer {i} 应为 SparseMoe");
        }
        assert!(c.layer_types.iter().all(|t| t == "full_attention"));
        assert_eq!(c.scoring_func, "softmax");
        assert!(!c.use_routing_bias);
    }

    #[test]
    fn snapshot_minimax_m2_5() {
        let Some(c) = load_ckpt("MiniMax-M2.5") else {
            return;
        };
        assert_eq!(c.model_type, "minimax_m2");
        assert_eq!(c.num_hidden_layers, 62);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.rotary_dim, 64); // JSON 显式给 rotary_dim（旧 rotary_dim: Option）
        assert_eq!(c.rope_theta, 5_000_000);
        // 缺口 G0：`num_local_experts` 经 alias 接上 → 62 层全 SparseMoe。
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.decoder_sparse_step, 1);
        assert_eq!(c.moe_intermediate_size, 1536); // 无 moe_intermediate_size 键
        for i in 0..c.num_hidden_layers {
            assert!(is_sparse(&c, i), "layer {i} 应为 SparseMoe");
        }
        assert!(c.layer_types.iter().all(|t| t == "full_attention"));
        // 缺口：取消家族兜底 —— ckpt 显式写了这两键，故不依赖任何 profile。
        assert_eq!(c.scoring_func, "sigmoid");
        assert!(c.use_routing_bias);
    }

    #[test]
    fn snapshot_qwen3_8_27b_is_hybrid_with_dense_ffn() {
        let Some(c) = load_ckpt("Qwen3.8-27B") else {
            return;
        };
        // 嵌套 text_config 为准 → 子配置的 model_type。
        assert_eq!(c.model_type, "qwen3_5_text");
        assert_eq!(c.num_hidden_layers, 64);
        assert_eq!(c.head_dim, 256);
        assert_eq!(c.num_key_value_heads, 4);
        // qwen3_5 家族默认：partial_rotary_factor 0.25 → 256 × 0.25 = 64。
        assert_eq!(c.partial_rotary_factor, 0.25);
        assert_eq!(c.rotary_dim, 64);
        // 顶层无 rope_theta，实际值藏在 rope_parameters 内。
        assert_eq!(c.rope_theta, 10_000_000);
        // 显式 64 项清单：idx 3/7/…/63 为 full，其余 linear（不重生成）。
        assert_eq!(c.layer_types.len(), 64);
        for (i, t) in c.layer_types.iter().enumerate() {
            if (i + 1) % 4 == 0 {
                assert_eq!(t, "full_attention", "layer {i}");
            } else {
                assert_eq!(t, "linear_attention", "layer {i}");
            }
        }
        // Python 事实：config.json 无任何 MoE 键 → 64 层 FFN 全 dense
        //（`self.mlp = Qwen3_5MLP(config, config.intermediate_size)` 无条件）。
        assert_eq!(c.num_experts, 0);
        for i in 0..c.num_hidden_layers {
            assert!(!is_sparse(&c, i), "layer {i} 应为 dense");
        }
        assert_eq!(c.intermediate_size, 17408);
        // 缺口 G2：输出门控的数据已收进配置，实现在层代码。
        assert!(c.attn_output_gate);
        assert_eq!(c.output_gate_type, "swish");
    }
}
