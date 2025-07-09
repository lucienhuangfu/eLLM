use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

// use core::convert::From;
use crate::kernel::generic::from_f32::FromF32;

use serde::de::{self, Deserializer, Visitor};
use std::f16;


#[derive(Debug, Serialize, Deserialize, Clone)]
struct _Config {
    pub _name_or_path: String,
    pub architectures: Vec<String>,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pretraining_tp: usize,
    pub rms_norm_eps: f32,
    pub rope_scaling: Option<HashMap<String, String>>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub _name_or_path: String,
    pub architectures: Vec<String>,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pretraining_tp: usize,
    pub rms_norm_eps: f32,
    pub rope_scaling: Option<HashMap<String, String>>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,

    pub attention_head_size: usize,
    pub vocab_size: usize,
    pub batch_size: usize,
    pub multiple_of: usize,
}
impl Config
// where
//     T: FromF32 + for<'de> Deserialize<'de>,
{
    pub fn new() -> Self {
        // let inverse_sqrt_head = f32::sqrt(attention_head_size as f32).recip();
        let hidden_size = 8192;
        let num_attention_heads = 64;
        let attention_head_size = hidden_size / num_attention_heads;
        Config {
            _name_or_path: String::new(),
            architectures: vec![],
            bos_token_id: 1,
            eos_token_id: 2,
            hidden_act: String::new(),
            hidden_size: hidden_size,
            initializer_range: 0.02,
            intermediate_size: 11008,
            num_hidden_layers: 80,
            num_attention_heads: num_attention_heads,
            max_position_embeddings: 4,
            model_type: String::from("llama"),
            num_key_value_heads: 8,
            pretraining_tp: 1,
            rms_norm_eps: 1e-6,
            rope_scaling: None,
            tie_word_embeddings: false,
            torch_dtype: String::from("float32"),
            transformers_version: String::new(),
            use_cache: true,
            vocab_size: 32000,

            attention_head_size: attention_head_size,
            batch_size: 3,
            multiple_of: 256,
        }
    }

    pub fn load_model_config<P: AsRef<Path>>(&mut self, filename: P) {
        let file = File::open(filename).unwrap();
        let reader = BufReader::new(file);
        let tmp_config: _Config = serde_json::from_reader(reader).unwrap();
        let attention_head_size = tmp_config.hidden_size / tmp_config.num_attention_heads;

        self._name_or_path = tmp_config._name_or_path;
        self.architectures = tmp_config.architectures;
        self.bos_token_id = tmp_config.bos_token_id;
        self.eos_token_id = tmp_config.eos_token_id;
        self.hidden_act = tmp_config.hidden_act;
        self.hidden_size = tmp_config.hidden_size;
        self.initializer_range = tmp_config.initializer_range;
        self.intermediate_size = tmp_config.intermediate_size;
        self.max_position_embeddings = tmp_config.max_position_embeddings;
        self.model_type = tmp_config.model_type;
        self.num_attention_heads = tmp_config.num_attention_heads;
        self.num_hidden_layers = tmp_config.num_hidden_layers;
        self.num_key_value_heads = tmp_config.num_key_value_heads;
        self.pretraining_tp = tmp_config.pretraining_tp;
        self.rms_norm_eps = tmp_config.rms_norm_eps;
        self.rope_scaling = tmp_config.rope_scaling;
        self.tie_word_embeddings = tmp_config.tie_word_embeddings;
        self.torch_dtype = tmp_config.torch_dtype;
        self.transformers_version = tmp_config.transformers_version;
        self.use_cache = tmp_config.use_cache;
        self.vocab_size = tmp_config.vocab_size;
        self.attention_head_size = attention_head_size;
    }

    pub fn load_compile_config<P: AsRef<Path>>(&mut self, filename: P) {
        let file = File::open(filename).unwrap();
        let reader = BufReader::new(file);
        let json: serde_json::Value = serde_json::from_reader(reader).unwrap();

        self.batch_size = json.get("batch_size").unwrap().as_u64().unwrap() as usize;
        self.multiple_of = json.get("multiple_of").unwrap().as_u64().unwrap() as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_file() {
        let mut config: Config = Config::new();
        config.load_model_config(r"models/Llama-2-7b-hf/config.json");
        config.load_compile_config(r"models/Llama-2-7b-hf.json");
        // println!("{:?}", config);
    }

}
