use std::collections::HashMap;
use std::f16;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Result};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};

// use serde_json;
// use std::io::{BufReader, Read};
// use std::arch::x86_64::bf16;
// use crate::common::config::Config;
// use crate::llama::model::Model;
// use crate::runtime::tensor::Tensor;

/// 在指定目录中查找所有safetensors文件
fn find_safetensors_files<P: AsRef<Path>>(model_dir: P) -> Result<Vec<std::path::PathBuf>> {
    let model_dir = model_dir.as_ref();
    let mut files = Vec::new();

    // 1. 优先检查常见的单文件命名
    let single_patterns = ["model.safetensors", "pytorch_model.safetensors"];
    for pattern in &single_patterns {
        let p = model_dir.join(pattern);
        if p.exists() {
            return Ok(vec![p]);
        }
    }

    // 2. 扫描目录查找分片文件
    let entries = std::fs::read_dir(model_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        // 简单检查扩展名
        if path.extension().map_or(false, |ext| ext == "safetensors") {
            files.push(path);
        }
    }

    if files.is_empty() {
        return Err(anyhow!(
            "No safetensors files found in {}",
            model_dir.display()
        ));
    }

    // 确保按文件名排序 (model-00001, model-00002...)
    files.sort();
    Ok(files)
}

/// 用于处理多文件safetensors模型的加载器
pub struct SafeTensorsLoader {
    pub model_files: Vec<String>,
    // config_path: String,
}

impl SafeTensorsLoader {
    /// 创建多文件safetensors加载器
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_files = find_safetensors_files(model_dir)?
            .into_iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        Ok(SafeTensorsLoader { model_files })
    }

    /// 加载所有权重文件
    pub fn load_all_weights_f16(&self) -> Result<HashMap<String, Vec<f16>>> {
        // 预估容量以减少重新哈希
        let mut all_weights = HashMap::with_capacity(512);

        for model_file in &self.model_files {
            let file = File::open(model_file)?;
            // 使用 mmap 避免将整个文件读入堆内存
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let safetensors = SafeTensors::deserialize(&mmap)?;

            for (name, tensor_view) in safetensors.tensors() {
                let data = match tensor_view.dtype() {
                    Dtype::F16 => tensor_view
                        .data()
                        .chunks_exact(2)
                        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect(),
                    Dtype::F32 => tensor_view
                        .data()
                        .chunks_exact(4)
                        .map(|chunk| {
                            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            val as f16
                        })
                        .collect(),
                    Dtype::BF16 => {
                        tensor_view
                            .data()
                            .chunks_exact(2)
                            .map(|chunk| {
                                let val_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                                // BF16 is upper 16 bits of F32
                                let val_f32 = f32::from_bits((val_u16 as u32) << 16);
                                val_f32 as f16
                            })
                            .collect()
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
        }

        Ok(all_weights)
    }

    /// 将每一层Transformer的所有MoE专家参数（w1, w2, w3等）合并成对应的三个大矩阵
    /// 最终生成的键名格式: model.layers.{i}.mlp.experts.{proj}.weight
    /// 例如: model.layers.0.mlp.experts.gate_proj.weight
    /// 数据排列顺序: 按专家索引排序 [Expert0_proj, Expert1_proj, ...]
    pub fn merge_moe(&self, weights: &mut HashMap<String, Vec<f16>>) -> Result<()> {
        // 优化：使用 Vec 存储元数据进行排序
        // (prefix, suffix, expert_idx, original_key)
        let mut moe_keys: Vec<(String, String, usize, String)> = Vec::new();

        // 1. 扫描并收集 MoE 相关键
        for key in weights.keys() {
            if key.contains("mlp.experts.") {
                // key example: model.layers.0.mlp.experts.0.w1.weight
                if let Some((prefix, rest)) = key.split_once("experts.") {
                    if let Some((expert_idx_str, suffix)) = rest.split_once('.') {
                        if let Ok(expert_idx) = expert_idx_str.parse::<usize>() {
                            moe_keys.push((
                                prefix.to_string(),
                                suffix.to_string(),
                                expert_idx,
                                key.clone(),
                            ));
                        }
                    }
                }
            }
        }

        // 2. 排序：先按层(prefix)，再按后缀(proj type)，最后按专家索引
        moe_keys.sort_by(|a, b| {
            let prefix_cmp = a.0.cmp(&b.0);
            if prefix_cmp != std::cmp::Ordering::Equal {
                return prefix_cmp;
            }
            let suffix_cmp = a.1.cmp(&b.1);
            if suffix_cmp != std::cmp::Ordering::Equal {
                return suffix_cmp;
            }
            a.2.cmp(&b.2)
        });

        // 3. 按层和后缀分组并合并
        let mut i = 0;
        while i < moe_keys.len() {
            let current_prefix = &moe_keys[i].0;
            let current_suffix = &moe_keys[i].1;
            let mut j = i;

            // 找到属于同一层且同一投影类型的所有条目
            while j < moe_keys.len()
                && &moe_keys[j].0 == current_prefix
                && &moe_keys[j].1 == current_suffix
            {
                j += 1;
            }

            let layer_items = &moe_keys[i..j];

            // 计算合并后的总大小
            let total_len: usize = layer_items
                .iter()
                .filter_map(|(_, _, _, k)| weights.get(k).map(|v| v.len()))
                .sum();

            if total_len > 0 {
                let mut merged_data = Vec::with_capacity(total_len);

                // 按顺序提取并合并数据
                for (_, _, _, key) in layer_items {
                    if let Some(data) = weights.remove(key) {
                        merged_data.extend(data);
                    }
                }

                // 插入合并后的新键: prefix + "experts." + suffix
                let new_key = format!("{}experts.{}", current_prefix, current_suffix);
                weights.insert(new_key, merged_data);
            }

            i = j;
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
            vec![1.0f16],
        );
        weights.insert(
            format!("{}experts.0.gate_proj.weight", prefix),
            vec![2.0f16],
        );
        weights.insert(format!("{}experts.0.up_proj.weight", prefix), vec![3.0f16]);

        // Expert 1
        weights.insert(
            format!("{}experts.1.down_proj.weight", prefix),
            vec![4.0f16],
        );
        weights.insert(
            format!("{}experts.1.gate_proj.weight", prefix),
            vec![5.0f16],
        );
        weights.insert(format!("{}experts.1.up_proj.weight", prefix), vec![6.0f16]);

        loader.merge_moe(&mut weights).unwrap();

        // 验证 down_proj
        let down_key = format!("{}experts.down_proj.weight", prefix);
        assert!(weights.contains_key(&down_key));
        let down_merged = weights.get(&down_key).unwrap();
        assert_eq!(down_merged.len(), 2);
        assert_eq!(down_merged[0], 1.0f16);
        assert_eq!(down_merged[1], 4.0f16);

        // 验证 gate_proj
        let gate_key = format!("{}experts.gate_proj.weight", prefix);
        assert!(weights.contains_key(&gate_key));
        let gate_merged = weights.get(&gate_key).unwrap();
        assert_eq!(gate_merged.len(), 2);
        assert_eq!(gate_merged[0], 2.0f16);
        assert_eq!(gate_merged[1], 5.0f16);

        // 验证 up_proj
        let up_key = format!("{}experts.up_proj.weight", prefix);
        assert!(weights.contains_key(&up_key));
        let up_merged = weights.get(&up_key).unwrap();
        assert_eq!(up_merged.len(), 2);
        assert_eq!(up_merged[0], 3.0f16);
        assert_eq!(up_merged[1], 6.0f16);
    }
}

