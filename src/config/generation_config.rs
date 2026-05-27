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
    pub min_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub do_sample: Option<bool>,
    #[serde(default)]
    pub num_beams: Option<usize>,
    #[serde(default)]
    #[serde(alias = "eos_token_id")]
    pub eos_token_id_list: Option<Vec<usize>>,
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    #[serde(default)]
    pub bos_token_id: Option<usize>,
    #[serde(default)]
    pub thread_num: Option<usize>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(flatten)]
    pub other: HashMap<String, Value>,
}

impl GenerationConfig {
    #[inline]
    pub fn align_top_k(top_k: usize, simd_width: usize) -> usize {
        if simd_width == 0 {
            top_k
        } else {
            top_k.div_ceil(simd_width) * simd_width
        }
    }

    #[inline]
    pub fn top_k_simd(&self, default_top_k: usize, simd_width: usize) -> usize {
        let top_k = self.top_k.unwrap_or(default_top_k);
        Self::align_top_k(top_k, simd_width)
    }

    #[inline]
    pub fn thread_num(&self) -> usize {
        self.thread_num
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1)
    }

    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let config: GenerationConfig = serde_json::from_reader(reader)?;
        Ok(config)
    }
}
