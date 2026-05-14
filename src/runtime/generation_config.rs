use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

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
    pub eos_token_id: Option<usize>,
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
}
