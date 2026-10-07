use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

/// Raw Hugging Face `config.json` shape: strings, numbers, and options only — no runtime enums.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct HfConfig {
    #[serde(default)]
    pub(crate) architectures: Vec<String>,
    #[serde(default)]
    pub(crate) attention_dropout: f32,
    #[serde(default = "default_decoder_sparse_step")]
    pub(crate) decoder_sparse_step: usize,
    #[serde(default)]
    pub(crate) eos_token_id: usize,
    pub(crate) head_dim: Option<usize>,
    #[serde(default)]
    pub(crate) hidden_act: String,
    #[serde(default)]
    pub(crate) hidden_size: usize,
    #[serde(default)]
    pub(crate) initializer_range: f32,
    pub(crate) intermediate_size: Option<usize>,
    #[serde(default)]
    pub(crate) max_position_embeddings: usize,
    pub(crate) max_window_layers: Option<usize>,
    #[serde(default)]
    pub(crate) mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub(crate) model_type: String,
    pub(crate) moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub(crate) norm_topk_prob: bool,
    #[serde(default)]
    pub(crate) num_attention_heads: usize,
    pub(crate) num_experts: Option<usize>,
    pub(crate) num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub(crate) output_router_logits: bool,
    #[serde(default)]
    pub(crate) qkv_bias: bool,
    #[serde(default)]
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_scaling: Option<HashMap<String, Value>>,
    pub(crate) rope_theta: Option<usize>,
    pub(crate) rotary_dim: Option<usize>,
    pub(crate) scoring_func: Option<String>,
    #[serde(default)]
    pub(crate) router_aux_loss_coef: f32,
    pub(crate) shared_experts_intermediate_size: Option<usize>,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) use_routing_bias: Option<bool>,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
    #[serde(default)]
    pub(crate) torch_dtype: String,
    #[serde(default)]
    pub(crate) transformers_version: String,
    #[serde(default)]
    pub(crate) use_cache: bool,
    #[serde(default)]
    pub(crate) use_qk_norm: bool,
    #[serde(default)]
    pub(crate) use_sliding_window: bool,
    #[serde(default)]
    pub(crate) vocab_size: usize,
    pub(crate) layer_types: Option<Vec<String>>,
    // GatedDeltaNet (linear attention) block dims; only present for hybrid
    // models whose `layer_types` contain `linear_attention`.
    pub(crate) linear_num_key_heads: Option<usize>,
    pub(crate) linear_num_value_heads: Option<usize>,
    pub(crate) linear_key_head_dim: Option<usize>,
    pub(crate) linear_value_head_dim: Option<usize>,
    pub(crate) linear_conv_kernel_dim: Option<usize>,
    // ── 多模态嵌套配置（qwen3_5：顶层 config.json 内嵌 text_config / vision_config）──
    // 扁平模型（qwen3_moe 等）两者皆为 None；嵌套模型的顶层扁平字段多为默认值，
    // 真实文本参数位于 text_config 内，故 qwen3_5 优先读取嵌套子配置。
    pub(crate) text_config: Option<Box<HfConfig>>,
    pub(crate) vision_config: Option<HfVisionConfig>,
    // 多模态特殊 token id，仅顶层 config 提供（对应 Qwen3_5Config 字段）。
    pub(crate) image_token_id: Option<usize>,
    pub(crate) video_token_id: Option<usize>,
    pub(crate) vision_start_token_id: Option<usize>,
    pub(crate) vision_end_token_id: Option<usize>,
    // 部分旋转因子：rotary_dim = head_dim × partial_rotary_factor（qwen3_5 默认 0.25）。
    pub(crate) partial_rotary_factor: Option<f32>,
    // 全注意力间隔：layer_types 缺省时每隔 interval 层插入一个 full_attention（qwen3_5 默认 4）。
    pub(crate) full_attention_interval: Option<usize>,
}

/// 视觉编码器（ViT）的原始 `config.json` 片段，对应 `Qwen3_5VisionConfig`。
/// 字段全部为 `Option`：与 `HfConfig` 一致，仅承载“原始 JSON 里出现的值”，
/// py 端 dataclass 默认值（depth=27、hidden_size=1152…）留到
/// `model_family::qwen3_5::VisionConfig::from_hf` 再应用，以便区分“缺失”与“显式 0”。
/// 注：`patch_size` / `temporal_patch_size` 在 py 里可为 int|list|tuple，Qwen 系列
/// config.json 实际 dump 为标量，故此处按标量 `usize` 处理。
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct HfVisionConfig {
    pub(crate) depth: Option<usize>,
    pub(crate) hidden_size: Option<usize>,
    pub(crate) hidden_act: Option<String>,
    pub(crate) intermediate_size: Option<usize>,
    pub(crate) num_heads: Option<usize>,
    pub(crate) in_channels: Option<usize>,
    pub(crate) patch_size: Option<usize>,
    pub(crate) spatial_merge_size: Option<usize>,
    pub(crate) temporal_patch_size: Option<usize>,
    pub(crate) out_hidden_size: Option<usize>,
    pub(crate) num_position_embeddings: Option<usize>,
    pub(crate) initializer_range: Option<f32>,
    pub(crate) rope_parameters: Option<HashMap<String, Value>>,
}

fn default_decoder_sparse_step() -> usize {
    1
}

impl HfConfig {
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_reader(reader)?)
    }

    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        Self::from_reader(BufReader::new(file))
    }
}
