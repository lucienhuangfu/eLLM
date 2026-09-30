use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelName {
    Qwen,
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
            "llama" => ModelName::Llama,
            "mixtral" => ModelName::Mixtral,
            "minimax" => ModelName::MiniMax,
            "minimax_m2" | "minimax-m2" | "minimax_m2.5" | "minimax-m2.5" => ModelName::MiniMaxM2,
            _ => ModelName::Unknown(model_type),
        }
    }
}
