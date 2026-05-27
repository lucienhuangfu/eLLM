use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

pub const MAX_EOS_TOKEN_IDS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EosTokenIds {
    ids: [usize; MAX_EOS_TOKEN_IDS],
    len: usize,
}

impl Default for EosTokenIds {
    fn default() -> Self {
        Self {
            ids: [0; MAX_EOS_TOKEN_IDS],
            len: 0,
        }
    }
}

impl EosTokenIds {
    pub fn from_slice(ids: &[usize]) -> Self {
        let mut eos = Self::default();
        let len = ids.len().min(MAX_EOS_TOKEN_IDS);
        eos.ids[..len].copy_from_slice(&ids[..len]);
        eos.len = len;
        eos
    }

    pub fn single(id: usize) -> Self {
        Self::from_slice(&[id])
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.ids[..self.len]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn primary(&self) -> usize {
        self.as_slice().first().copied().unwrap_or(0)
    }

    pub fn contains(&self, token_id: usize) -> bool {
        self.as_slice().contains(&token_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TokenIdOrIds {
    One(usize),
    Many(Vec<usize>),
}

impl Default for TokenIdOrIds {
    fn default() -> Self {
        Self::Many(Vec::new())
    }
}

impl TokenIdOrIds {
    pub fn as_slice(&self) -> &[usize] {
        match self {
            Self::One(id) => std::slice::from_ref(id),
            Self::Many(ids) => ids.as_slice(),
        }
    }

    pub fn first(&self) -> Option<usize> {
        self.as_slice().first().copied()
    }

    pub fn to_eos_token_ids(&self) -> EosTokenIds {
        EosTokenIds::from_slice(self.as_slice())
    }
}

impl From<usize> for TokenIdOrIds {
    fn from(value: usize) -> Self {
        Self::One(value)
    }
}

impl From<Vec<usize>> for TokenIdOrIds {
    fn from(value: Vec<usize>) -> Self {
        Self::Many(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    #[serde(default)]
    pub max_length: Option<usize>,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub do_sample: Option<bool>,
    #[serde(default)]
    pub num_beams: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Option<TokenIdOrIds>,
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    #[serde(default)]
    pub bos_token_id: Option<usize>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(flatten)]
    pub other: HashMap<String, Value>,
}

impl GenerationConfig {
    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let config: GenerationConfig = serde_json::from_reader(reader)?;
        Ok(config)
    }

    pub fn do_sample(&self) -> bool {
        false
    }

    pub fn effective_top_k(&self, default_top_k: usize) -> usize {
        if self.do_sample() {
            self.top_k.unwrap_or(default_top_k).max(1)
        } else {
            1
        }
    }

    pub fn effective_temperature(&self, default_temperature: f32) -> f32 {
        if self.do_sample() {
            self.temperature
                .map(|temperature| temperature as f32)
                .filter(|temperature| *temperature > 0.0)
                .unwrap_or(default_temperature)
        } else {
            1.0
        }
    }

    pub fn eos_token_ids(&self) -> Option<EosTokenIds> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdOrIds::to_eos_token_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationConfig, TokenIdOrIds};

    #[test]
    fn parses_single_and_multiple_eos_token_ids() {
        let single: GenerationConfig = serde_json::from_str(
            r#"{
                "do_sample": true,
                "eos_token_id": 200020,
                "top_k": 40
            }"#,
        )
        .unwrap();
        assert_eq!(
            single.eos_token_ids().map(|ids| ids.as_slice().to_vec()),
            Some(vec![200020])
        );

        let multiple: GenerationConfig = serde_json::from_str(
            r#"{
                "do_sample": true,
                "eos_token_id": [151645, 151643],
                "top_k": 20
            }"#,
        )
        .unwrap();
        assert_eq!(
            multiple.eos_token_id,
            Some(TokenIdOrIds::Many(vec![151645, 151643]))
        );
    }

    #[test]
    fn greedy_generation_ignores_sampling_knobs() {
        let config: GenerationConfig = serde_json::from_str(
            r#"{
                "do_sample": false,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95
            }"#,
        )
        .unwrap();

        assert_eq!(config.effective_top_k(8), 1);
        assert_eq!(config.effective_temperature(0.7), 1.0);
    }

    #[test]
    fn sampled_generation_is_forced_to_greedy() {
        let config: GenerationConfig = serde_json::from_str(
            r#"{
                "do_sample": true,
                "temperature": 0.6,
                "top_k": 20
            }"#,
        )
        .unwrap();

        assert_eq!(config.effective_top_k(8), 1);
        assert_eq!(config.effective_temperature(1.0), 1.0);
    }
}
