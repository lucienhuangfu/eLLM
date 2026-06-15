use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::BufReader, mem::size_of, path::Path};

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
    pub top_k_simd: Option<usize>,

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

    #[serde(default)]
    pub stop_token_ids: Option<Vec<usize>>,

    #[serde(default)]
    pub ignore_eos: Option<bool>,

    #[serde(default)]
    pub max_tokens: Option<usize>,

    #[serde(default)]
    pub min_tokens: Option<usize>,

    #[serde(default)]
    pub logprobs: Option<usize>,

    #[serde(default)]
    pub prompt_logprobs: Option<usize>,

    #[serde(default)]
    pub skip_special_tokens: Option<bool>,

    #[serde(default)]
    pub spaces_between_special_tokens: Option<bool>,

    #[serde(default)]
    pub include_stop_str_in_output: Option<bool>,

    #[serde(flatten)]
    pub extra_config: HashMap<String, Value>,
}

impl GenerationConfig {
    #[inline]
    pub fn avx512_simd_width<T>() -> usize {
        match size_of::<T>() {
            2 => 32,
            4 => 16,
            8 => 8,
            _ => 32,
        }
    }

    #[inline]
    pub fn align_top_k(top_k: usize, simd_width: usize) -> usize {
        if simd_width == 0 {
            top_k
        } else {
            top_k.div_ceil(simd_width) * simd_width
        }
    }

    #[inline]
    pub fn resolved_top_k_simd<T>(&self, default_top_k: usize) -> usize {
        let top_k = self.top_k.unwrap_or(default_top_k);
        let top_k_simd = self.top_k_simd.unwrap_or(top_k);
        let simd_width = Self::avx512_simd_width::<T>();
        Self::align_top_k(top_k_simd, simd_width)
    }

    #[inline]
    pub fn resolved_top_k_simd_static<T>(top_k: usize) -> usize {
        let simd_width = Self::avx512_simd_width::<T>();
        Self::align_top_k(top_k, simd_width)
    }

    #[inline]
    pub fn thread_num(&self) -> usize {
        self.thread_num
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1)
    }

    #[inline]
    pub fn resolved_temperature(&self) -> f64 {
        self.temperature.unwrap_or(1.0)
    }

    #[inline]
    pub fn resolved_top_p(&self) -> f64 {
        self.top_p.unwrap_or(1.0)
    }

    #[inline]
    pub fn resolved_repetition_penalty(&self) -> f64 {
        self.repetition_penalty.unwrap_or(1.0)
    }

    #[inline]
    pub fn resolved_do_sample(&self) -> bool {
        self.do_sample.unwrap_or(true)
    }

    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let config: GenerationConfig = serde_json::from_reader(reader)?;
        Ok(config)
    }
}
