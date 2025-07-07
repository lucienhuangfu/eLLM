use anyhow::{anyhow, Result};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde_json;
use std::collections::HashMap;
use std::f16;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::init::config::Config;
use crate::llama::model::Model;
use crate::ptensor::tensor::Tensor;

/// SafeTensors模型加载器
pub struct SafeTensorsModelLoader {
    /// 模型文件路径
    model_path: String,
    /// 配置文件路径
    config_path: String,
}

impl SafeTensorsModelLoader {
    /// 创建新的SafeTensors模型加载器
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // 查找safetensors文件
        let model_file = find_safetensors_file(model_dir)?;
        let config_file = model_dir.join("config.json");

        if !config_file.exists() {
            return Err(anyhow!("config.json not found in model directory"));
        }

        Ok(SafeTensorsModelLoader {
            model_path: model_file.to_string_lossy().to_string(),
            config_path: config_file.to_string_lossy().to_string(),
        })
    }

    /// 加载配置文件
    pub fn load_config(&self) -> Result<Config> {
        let file = File::open(&self.config_path)?;
        let reader = BufReader::new(file);
        let config: Config = serde_json::from_reader(reader)?;
        Ok(config)
    }

    /// 加载模型权重到HashMap
    pub fn load_weights_f16(&self) -> Result<HashMap<String, Vec<f16>>> {
        let file = File::open(&self.model_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let safetensors = SafeTensors::deserialize(&mmap)?;

        let mut weights = HashMap::new();

        for (name, tensor_view) in safetensors.tensors() {
            let data = match tensor_view.dtype() {
                Dtype::F16 => {
                    // 直接从f16数据转换
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
                    // 从f32转换到f16
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

            weights.insert(name.to_string(), data);
        }

        Ok(weights)
    }

    /// 加载完整的Llama3模型
    pub fn load_model<T>(&self) -> Result<Model<T>>
    where
        T: Copy
            + Default
            + std::ops::Sub<Output = T>
            + std::ops::Neg<Output = T>
            + crate::kernel::generic::exp::Exp
            + crate::kernel::generic::neg_infinity::NegInfinity
            + crate::kernel::generic::sigmoid::Sigmoid<T>
            + crate::kernel::generic::sqrt::Sqrt
            + crate::kernel::generic::from_f32::FromF32,
    {
        let config = self.load_config()?;
        let weights = self.load_weights_f16()?;

        // 这里需要根据你的Model实现来构建模型
        // 由于Model的构造函数需要具体的参数，这里只是一个示例框架
        todo!("需要根据具体的Model实现来构建模型")
    }
}

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
pub struct MultiFileSafeTensorsLoader {
    model_files: Vec<String>,
    config_path: String,
}

impl MultiFileSafeTensorsLoader {
    /// 创建多文件safetensors加载器
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_file = model_dir.join("config.json");

        if !config_file.exists() {
            return Err(anyhow!("config.json not found in model directory"));
        }

        // 查找所有safetensors文件
        let mut model_files = Vec::new();
        let entries = std::fs::read_dir(model_dir)?;

        for entry in entries {
            let entry = entry?;
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            if file_name_str.ends_with(".safetensors") {
                model_files.push(entry.path().to_string_lossy().to_string());
            }
        }

        if model_files.is_empty() {
            return Err(anyhow!("No safetensors files found in model directory"));
        }

        // 排序确保正确的加载顺序
        model_files.sort();

        Ok(MultiFileSafeTensorsLoader {
            model_files,
            config_path: config_file.to_string_lossy().to_string(),
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
        }

        Ok(all_weights)
    }

    /// 加载配置
    pub fn load_config(&self) -> Result<Config> {
        let file = File::open(&self.config_path)?;
        let reader = BufReader::new(file);
        let config: Config = serde_json::from_reader(reader)?;
        Ok(config)
    }
}

/// 便民函数：从目录加载Llama3模型
pub fn load_llama3_from_safetensors<P: AsRef<Path>>(
    model_dir: P,
) -> Result<(Config, HashMap<String, Vec<f16>>)> {
    // 首先尝试单文件加载器
    if let Ok(loader) = SafeTensorsModelLoader::new(&model_dir) {
        let config = loader.load_config()?;
        let weights = loader.load_weights_f16()?;
        return Ok((config, weights));
    }

    // 如果单文件失败，尝试多文件加载器
    let loader = MultiFileSafeTensorsLoader::new(model_dir)?;
    let config = loader.load_config()?;
    let weights = loader.load_all_weights_f16()?;

    Ok((config, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_config() {
        // 这里可以添加测试代码
    }

    #[test]
    fn test_load_safetensors() {
        // 这里可以添加测试代码
    }
}
