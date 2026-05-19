use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamily {
    Qwen,
    Llama,
    Mixtral,
    MiniMax,
    MiniMaxM2,
    Unknown(String),
}

impl ModelFamily {
    pub fn parse(model_type: &str) -> Self {
        let model_type = model_type.to_ascii_lowercase();
        match model_type.as_str() {
            "qwen2" | "qwen2_moe" | "qwen3" | "qwen3_moe" => ModelFamily::Qwen,
            "llama" => ModelFamily::Llama,
            "mixtral" => ModelFamily::Mixtral,
            "minimax" => ModelFamily::MiniMax,
            "minimax_m2" | "minimax-m2" | "minimax_m2.5" | "minimax-m2.5" => ModelFamily::MiniMaxM2,
            _ => ModelFamily::Unknown(model_type),
        }
    }
}
