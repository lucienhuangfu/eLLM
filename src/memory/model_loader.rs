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
    pub(crate) model_files: Vec<String>,
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
                println!(
                    "Found safetensors file: {}",
                    entry.path().to_string_lossy().to_string()
                );
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

    /// 将每一层Transformer的所有MoE专家参数（w1, w2, w3等）合并成一个巨大的参数矩阵
    /// 最终生成的键名格式: model.layers.{i}.block_sparse_moe.experts.weight
    /// 数据排列顺序: 按参数名称(w1, w2...)排序，同名参数按专家索引排序
    /// 结果布局: [All_Experts_w1, All_Experts_w2, All_Experts_w3] (假设w1, w2, w3为字母序)
    pub fn merge_moe(&self, weights: &mut HashMap<String, Vec<f16>>) -> Result<()> {
        // 存储每一层的MoE相关key
        // Key: layer_prefix (e.g., "model.layers.0.block_sparse_moe.")
        // Value: List of (expert_idx, suffix, original_key)
        let mut layer_groups: HashMap<String, Vec<(usize, String, String)>> = HashMap::new();

        // 1. 扫描并按层分组
        for key in weights.keys() {
            // mlp.experts
            if key.contains("mlp.experts.") {
                // key example: model.layers.0.block_sparse_moe.experts.0.w1.weight
                if let Some((prefix, rest)) = key.split_once("experts.") {
                    // prefix: model.layers.0.block_sparse_moe.
                    // rest: 0.w1.weight

                    if let Some((expert_idx_str, suffix)) = rest.split_once('.') {
                        // expert_idx_str: 0
                        // suffix: w1.weight

                        if let Ok(expert_idx) = expert_idx_str.parse::<usize>() {
                            layer_groups.entry(prefix.to_string()).or_default().push((
                                expert_idx,
                                suffix.to_string(),
                                key.clone(),
                            ));
                        }
                    }
                }
            }
        }

        // 2. 对每一层进行合并
        for (prefix, mut items) in layer_groups {
            // 排序规则：
            // 1. 先按参数类型后缀排序 (w1.weight, w2.weight, w3.weight)
            // 2. 再按专家索引排序 (0, 1, 2...)
            items.sort_by(|a, b| {
                let suffix_cmp = a.1.cmp(&b.1);
                if suffix_cmp == std::cmp::Ordering::Equal {
                    a.0.cmp(&b.0)
                } else {
                    suffix_cmp
                }
            });

            // 计算总大小
            let total_len: usize = items
                .iter()
                .filter_map(|(_, _, k)| weights.get(k).map(|v| v.len()))
                .sum();

            let mut merged_data = Vec::with_capacity(total_len);

            // 按顺序取出数据并合并，同时从原map中删除旧的key
            for (_, _, old_key) in &items {
                if let Some(data) = weights.remove(old_key) {
                    merged_data.extend(data);
                }
            }

            if !merged_data.is_empty() {
                // 新的键名: model.layers.{i}.mlp.experts.weight
                let new_key = format!("{}experts.weight", prefix);
                weights.insert(new_key, merged_data);
            }
        }

        Ok(())
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
    pub fn detailed_loading_example() {
        let model_dir = "D:/llama-3-chinese-8b-instruct-v3";
        if !std::path::Path::new(model_dir).exists() {
            println!("Model directory not found, skipping test.");
            return;
        }

        // 方法2：使用详细的加载器
        let loader = SafeTensorsLoader::new(model_dir).unwrap();

        // 分别加载配置和权重
        // let config = loader.load_config()?;
        let mut weights = loader.load_all_weights_f16().unwrap();

        // 尝试合并MoE权重
        // loader.merge_moe(&mut weights).unwrap();

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
                format!("{}.mlp.gate.weight", layer_prefix),
                format!("{}.mlp.experts.0.gate_proj.weight", layer_prefix),
                format!("{}.mlp.experts.0.up_proj.weight", layer_prefix),
                format!("{}.mlp.experts.0.down_proj.weight", layer_prefix),
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

    #[test]
    fn test_merge_moe_qwen() {
        // 模拟Qwen MoE结构
        let loader = SafeTensorsLoader {
            model_files: vec![],
        };

        let mut weights = HashMap::new();
        let prefix = "model.layers.0.mlp.";

        // 模拟2个专家，每个专家有 gate_proj, up_proj, down_proj
        // 这里的后缀排序顺序: down_proj (d), gate_proj (g), up_proj (u)

        // Expert 0
        weights.insert(
            format!("{}experts.0.down_proj.weight", prefix),
            vec![f16::from_f32(1.0)],
        );
        weights.insert(
            format!("{}experts.0.gate_proj.weight", prefix),
            vec![f16::from_f32(2.0)],
        );
        weights.insert(
            format!("{}experts.0.up_proj.weight", prefix),
            vec![f16::from_f32(3.0)],
        );

        // Expert 1
        weights.insert(
            format!("{}experts.1.down_proj.weight", prefix),
            vec![f16::from_f32(4.0)],
        );
        weights.insert(
            format!("{}experts.1.gate_proj.weight", prefix),
            vec![f16::from_f32(5.0)],
        );
        weights.insert(
            format!("{}experts.1.up_proj.weight", prefix),
            vec![f16::from_f32(6.0)],
        );

        loader.merge_moe(&mut weights).unwrap();

        let new_key = format!("{}experts.weight", prefix);
        assert!(weights.contains_key(&new_key));

        let merged = weights.get(&new_key).unwrap();

        // 验证顺序:
        // 1. down_proj (expert 0, then expert 1) -> 1.0, 4.0
        // 2. gate_proj (expert 0, then expert 1) -> 2.0, 5.0
        // 3. up_proj   (expert 0, then expert 1) -> 3.0, 6.0

        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

        assert_eq!(merged.len(), expected.len());
        for (i, &val) in merged.iter().enumerate() {
            assert_eq!(val.to_f32(), expected[i], "Mismatch at index {}", i);
        }
    }
}
