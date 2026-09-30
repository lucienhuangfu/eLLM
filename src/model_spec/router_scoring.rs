use serde::{Deserialize, Serialize};

use super::ModelName;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouterScoringKind {
    Softmax,
    Sigmoid,
}

impl RouterScoringKind {
    pub(crate) fn from_hf(scoring_func: Option<&str>, family: ModelName) -> Self {
        match scoring_func.map(|s| s.to_ascii_lowercase()) {
            Some(scoring) if scoring == "sigmoid" => RouterScoringKind::Sigmoid,
            Some(scoring) if scoring == "softmax" => RouterScoringKind::Softmax,
            _ => match family {
                ModelName::MiniMaxM2 => RouterScoringKind::Sigmoid,
                _ => RouterScoringKind::Softmax,
            },
        }
    }
}
