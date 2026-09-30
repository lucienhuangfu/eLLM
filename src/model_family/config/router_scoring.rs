use serde::{Deserialize, Serialize};

use super::model_family::ModelFamily;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouterScoringKind {
    Softmax,
    Sigmoid,
}

impl RouterScoringKind {
    pub(crate) fn from_hf(scoring_func: Option<&str>, family: ModelFamily) -> Self {
        match scoring_func.map(|s| s.to_ascii_lowercase()) {
            Some(scoring) if scoring == "sigmoid" => RouterScoringKind::Sigmoid,
            Some(scoring) if scoring == "softmax" => RouterScoringKind::Softmax,
            _ => match family {
                ModelFamily::MiniMaxM2 => RouterScoringKind::Sigmoid,
                _ => RouterScoringKind::Softmax,
            },
        }
    }
}
