use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelName {
    Qwen,
    Qwen3_5,
    Llama,
    Mixtral,
    MiniMax,
    MiniMaxM2,
    Unknown(String),
}

impl ModelName {
    pub fn parse(model_type: &str) -> Self {
        let model_type = model_type.to_ascii_lowercase();
        match model_type.as_str() {
            "qwen2" | "qwen2_moe" | "qwen3" | "qwen3_moe" => ModelName::Qwen,
            // qwen3_5：多模态（text + vision）+ 混合注意力（GatedDeltaNet 线性 / 全注意力）。
            // 顶层 model_type 为 "qwen3_5"，文本子配置为 "qwen3_5_text"。
            "qwen3_5" | "qwen3_5_text" => ModelName::Qwen3_5,
            "llama" => ModelName::Llama,
            "mixtral" => ModelName::Mixtral,
            "minimax" => ModelName::MiniMax,
            "minimax_m2" | "minimax-m2" | "minimax_m2.5" | "minimax-m2.5" => ModelName::MiniMaxM2,
            _ => ModelName::Unknown(model_type),
        }
    }
}
