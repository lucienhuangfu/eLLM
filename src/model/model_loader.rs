
use std::collections::HashMap;
use std::f16;
// use std::arch::x86_64::bf16;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{anyhow, Result};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde_json;

// use crate::init::config::Config;
// use crate::llama::model::Model;
// use crate::ptensor::tensor::Tensor;

/// 在指定目录中查找safetensors文件
fn find_safetensors_file<P: AsRef<Path>>(model_dir: P) -> Result<std::path::PathBuf> {
    let model_dir = model_dir.as_ref();

    // 常见的safetensors文件名模式
    let patterns = [
        "model.safetensors",
        "pytorch_model.safetensors",
        "model-00001-of-00001.safetensors",
    ];

    // 首先尝试单文件模式
    for pattern in &patterns {
        let file_path = model_dir.join(pattern);
        if file_path.exists() {
            return Ok(file_path);
        }
    }

    // 如果没找到单文件，查找分片文件
    let entries = std::fs::read_dir(model_dir)?;
    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        if file_name_str.starts_with("model-") && file_name_str.ends_with(".safetensors") {
            // 找到第一个分片文件，返回它
            // 注意：如果是多文件模式，可能需要更复杂的逻辑来处理所有分片
            return Ok(entry.path());
        }
    }

    Err(anyhow!("No safetensors file found in the model directory"))
}

/// 用于处理多文件safetensors模型的加载器
pub struct SafeTensorsLoader {
    model_files: Vec<String>,
    // config_path: String,
}

impl SafeTensorsLoader {
    /// 创建多文件safetensors加载器
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        /* 
        let config_file = model_dir.join("config.json");

        if !config_file.exists() {
            return Err(anyhow!("config.json not found in model directory"));
        }*/

        // 查找所有safetensors文件
        let mut model_files = Vec::new();
        let entries = std::fs::read_dir(model_dir)?;

        for entry in entries {
            let entry = entry?;
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            if file_name_str.ends_with(".safetensors") {
                println!("Found safetensors file: {}", entry.path().to_string_lossy().to_string());
                model_files.push(entry.path().to_string_lossy().to_string());
            }
        }

        if model_files.is_empty() {
            return Err(anyhow!("No safetensors files found in model directory"));
        }

        // 排序确保正确的加载顺序
        model_files.sort();

        Ok(SafeTensorsLoader {
            model_files,
            // config_path: config_file.to_string_lossy().to_string(),
        })
    }

    /// 加载所有权重文件
    pub fn load_all_weights_f16(&self) -> Result<HashMap<String, Vec<f16>>> {
        let mut all_weights = HashMap::new();
        
        for model_file in &self.model_files {
            let file = File::open(model_file)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let safetensors = SafeTensors::deserialize(&mmap)?;

            for (name, tensor_view) in safetensors.tensors() {
                let data = match tensor_view.dtype() {
                    Dtype::F16 => {
                        let raw_data = tensor_view.data();
                        let f16_data: Vec<f16> = raw_data
                            .chunks_exact(2)
                            .map(|chunk| {
                                let bytes = [chunk[0], chunk[1]];
                                f16::from_le_bytes(bytes)
                            })
                            .collect();
                        f16_data
                    }
                    Dtype::F32 => {
                        let raw_data = tensor_view.data();
                        let f32_data: Vec<f32> = raw_data
                            .chunks_exact(4)
                            .map(|chunk| {
                                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                                f32::from_le_bytes(bytes)
                            })
                            .collect();
                        f32_data.iter().map(|&x| x as f16).collect()
                    }
                    Dtype::BF16 => {
                        // 从BF16转换到std::f16
                        let raw_data = tensor_view.data();
                        let bf16_data: Vec<half::bf16> = raw_data
                            .chunks_exact(2)
                            .map(|chunk| {
                                let bytes = [chunk[0], chunk[1]];
                                half::bf16::from_le_bytes(bytes)
                            })
                            .collect();
                        bf16_data.iter().map(|&x| x.to_f32() as f16).collect()
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported tensor dtype: {:?}",
                            tensor_view.dtype()
                        ));
                    }
                };

                all_weights.insert(name.to_string(), data);
            }
            // break;
        }

        Ok(all_weights)
    }

    /*
    /// 加载配置
    pub fn load_config(&self) -> Result<Config> {
        let file = File::open(&self.config_path)?;
        let reader = BufReader::new(file);
        let config: Config = serde_json::from_reader(reader)?;
        Ok(config)
    } */
}

/// 便民函数：从目录加载Llama3模型

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_safetensors() {
        // 这里可以添加测试代码

        let torch_file = String::from("D:/llama-3-chinese-8b-instruct-v3");
        let loader = SafeTensorsLoader::new(&torch_file).unwrap();
        loader.load_all_weights_f16().unwrap();
    }

    #[test]
    /// 使用SafeTensorsModelLoader的详细示例
    pub fn detailed_loading_example()  {
        let model_dir = "D:/llama-3-chinese-8b-instruct-v3";

        // 方法2：使用详细的加载器
        let loader = SafeTensorsLoader::new(model_dir).unwrap();

        // 分别加载配置和权重
        // let config = loader.load_config()?;
        let weights = loader.load_all_weights_f16().unwrap();

        // 验证关键层的存在
        let expected_layers = [
            "model.embed_tokens.weight",
            // "model.norm.weight",
            // "lm_head.weight",
        ];

        for layer_name in &expected_layers {
            if weights.contains_key(*layer_name) {
                println!("✓ Found layer: {}", layer_name);
            } else {
                println!("✗ Missing layer: {}", layer_name);
            }
        }

        // 检查transformer层
        for i in 0..2 {
            let layer_prefix = format!("model.layers.{}", i);
            let attention_layers = [
                format!("{}.self_attn.q_proj.weight", layer_prefix),
                format!("{}.self_attn.k_proj.weight", layer_prefix),
                format!("{}.self_attn.v_proj.weight", layer_prefix),
                format!("{}.self_attn.o_proj.weight", layer_prefix),
                format!("{}.mlp.gate_proj.weight", layer_prefix),
                format!("{}.mlp.up_proj.weight", layer_prefix),
                format!("{}.mlp.down_proj.weight", layer_prefix),
                format!("{}.input_layernorm.weight", layer_prefix),
                format!("{}.post_attention_layernorm.weight", layer_prefix),
            ];

            let mut layer_complete = true;
            for layer_name in &attention_layers {
                if !weights.contains_key(layer_name) {
                    layer_complete = false;
                    break;
                }
            }

            if layer_complete {
                println!("✓ Layer {} complete", i);
            } else {
                println!("✗ Layer {} incomplete", i);
            }
        }

        println!("Model loading verification completed!");

        // Ok(())
    }
}
