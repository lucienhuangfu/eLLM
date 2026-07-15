use std::{collections::HashMap, f16, fs, mem, path::Path, thread};

use anyhow::{anyhow, Result};
use memmap2::MmapOptions;
use safetensors::tensor::TensorView;
use safetensors::Dtype;
use safetensors::SafeTensors;

pub trait FromSafetensors: Sized {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>>;
}

impl FromSafetensors for f16 {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>> {
        match tensor_view.dtype() {
            Dtype::F16 => copy_le_bytes_as_f16(tensor_view.data()),
            Dtype::F32 => Ok(tensor_view
                .data()
                .chunks_exact(4)
                .map(|chunk| {
                    let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    val as f16
                })
                .collect()),
            Dtype::BF16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| {
                    let val_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let val_f32 = f32::from_bits((val_u16 as u32) << 16);
                    val_f32 as f16
                })
                .collect()),
            _ => Err(anyhow!(
                "Unsupported tensor dtype for f16: {:?}",
                tensor_view.dtype()
            )),
        }
    }
}

#[inline]
fn copy_le_bytes_as_f16(data: &[u8]) -> Result<Vec<f16>> {
    if data.len() % mem::size_of::<f16>() != 0 {
        return Err(anyhow!(
            "Invalid F16 tensor byte length: {} is not divisible by {}",
            data.len(),
            mem::size_of::<f16>()
        ));
    }

    #[cfg(target_endian = "little")]
    {
        let len = data.len() / mem::size_of::<f16>();
        let mut out = Vec::<f16>::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr().cast::<u8>(), data.len());
            out.set_len(len);
        }
        Ok(out)
    }

    #[cfg(not(target_endian = "little"))]
    {
        Ok(data
            .chunks_exact(2)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }
}

impl FromSafetensors for f32 {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>> {
        match tensor_view.dtype() {
            Dtype::F16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]) as f32)
                .collect()),
            Dtype::F32 => Ok(tensor_view
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            Dtype::BF16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| {
                    let val_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32::from_bits((val_u16 as u32) << 16)
                })
                .collect()),
            _ => Err(anyhow!(
                "Unsupported tensor dtype for f32: {:?}",
                tensor_view.dtype()
            )),
        }
    }
}

impl FromSafetensors for f64 {
    fn from_tensor_view(tensor_view: &TensorView) -> Result<Vec<Self>> {
        match tensor_view.dtype() {
            Dtype::F16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]) as f64)
                .collect()),
            Dtype::F32 => Ok(tensor_view
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
                .collect()),
            Dtype::BF16 => Ok(tensor_view
                .data()
                .chunks_exact(2)
                .map(|chunk| {
                    let val_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32::from_bits((val_u16 as u32) << 16) as f64
                })
                .collect()),
            _ => Err(anyhow!(
                "Unsupported tensor dtype for f64: {:?}",
                tensor_view.dtype()
            )),
        }
    }
}

pub struct SafeTensorsLoader {
    pub model_files: Vec<String>,
}

impl SafeTensorsLoader {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let single_patterns = ["model.safetensors", "pytorch_model.safetensors"];
        for pattern in &single_patterns {
            let p = model_dir.join(pattern);
            if p.exists() {
                return Ok(SafeTensorsLoader {
                    model_files: vec![p.to_string_lossy().to_string()],
                });
            }
        }

        let mut files = Vec::new();
        let entries = std::fs::read_dir(model_dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
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

        files.sort();
        Ok(SafeTensorsLoader {
            model_files: files
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect(),
        })
    }

    fn load_file_weights<T: FromSafetensors>(model_file: &str) -> Result<HashMap<String, Vec<T>>> {
        let file = fs::File::open(model_file)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let safetensors = SafeTensors::deserialize(&mmap)?;

        let mut weights = HashMap::with_capacity(safetensors.tensors().len());
        for (name, tensor_view) in safetensors.tensors() {
            let data = T::from_tensor_view(&tensor_view)?;
            weights.insert(name.to_string(), data);
        }

        Ok(weights)
    }

    pub fn load_all_weights<T: FromSafetensors>(&self) -> Result<HashMap<String, Vec<T>>> {
        let mut all_weights = HashMap::with_capacity(512);

        for model_file in &self.model_files {
            all_weights.extend(Self::load_file_weights::<T>(model_file)?);
        }

        Ok(all_weights)
    }

    pub fn load_all_weights_parallel<T>(&self) -> Result<HashMap<String, Vec<T>>>
    where
        T: FromSafetensors + Send,
    {
        let thread_count = std::env::var("ELLM_LOAD_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            })
            .min(self.model_files.len())
            .max(1);

        let mut all_weights = HashMap::with_capacity(512);

        for chunk in self.model_files.chunks(thread_count) {
            let file_maps = thread::scope(|scope| {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|model_file| {
                        scope.spawn(move || Self::load_file_weights::<T>(model_file.as_str()))
                    })
                    .collect();

                handles
                    .into_iter()
                    .map(|handle| {
                        handle
                            .join()
                            .map_err(|_| anyhow!("parallel safetensors loader thread panicked"))?
                    })
                    .collect::<Result<Vec<_>>>()
            })?;

            for file_weights in file_maps {
                all_weights.extend(file_weights);
            }
        }

        Ok(all_weights)
    }

    pub fn load_all_weights_f16(&self) -> Result<HashMap<String, Vec<f16>>> {
        self.load_all_weights::<f16>()
    }

    pub fn load_all_weights_f16_parallel(&self) -> Result<HashMap<String, Vec<f16>>> {
        self.load_all_weights_parallel::<f16>()
    }

    pub fn merge_moe<T>(&self, weights: &mut HashMap<String, Vec<T>>) -> Result<()> {
        let mut moe_keys: Vec<(String, String, usize, String)> = Vec::new();

        for key in weights.keys() {
            if key.contains("mlp.experts.") {
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

        moe_keys.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut i = 0;
        while i < moe_keys.len() {
            let current_prefix = &moe_keys[i].0;
            let current_suffix = &moe_keys[i].1;
            let mut j = i;

            while j < moe_keys.len()
                && &moe_keys[j].0 == current_prefix
                && &moe_keys[j].1 == current_suffix
            {
                j += 1;
            }

            let layer_items = &moe_keys[i..j];
            let total_len: usize = layer_items
                .iter()
                .filter_map(|(_, _, _, k)| weights.get(k).map(|v| v.len()))
                .sum();

            if total_len > 0 {
                let mut merged_data = Vec::with_capacity(total_len);
                for (_, _, _, key) in layer_items {
                    if let Some(data) = weights.remove(key) {
                        merged_data.extend(data);
                    }
                }

                let new_key = format!("{}experts.{}", current_prefix, current_suffix);
                weights.insert(new_key, merged_data);
            }

            i = j;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_le_bytes_as_f16_preserves_values() {
        let values = [1.0f16, -2.5f16, 0.0f16, f16::INFINITY];
        let mut bytes = Vec::with_capacity(values.len() * mem::size_of::<f16>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let copied = copy_le_bytes_as_f16(&bytes).unwrap();
        assert_eq!(copied, values);
    }

    #[test]
    fn copy_le_bytes_as_f16_rejects_odd_byte_count() {
        let err = copy_le_bytes_as_f16(&[0, 1, 2]).unwrap_err();
        assert!(err.to_string().contains("Invalid F16 tensor byte length"));
    }

    #[test]
    #[ignore = "Requires models/Qwen3-0.6B to be present"]
    fn test_load_qwen3_06b_f16() {
        let loader = SafeTensorsLoader::new("models/Qwen3-0.6B").unwrap();
        let weights = loader.load_all_weights::<f16>().unwrap();
        assert!(!weights.is_empty());
    }

    #[test]
    #[ignore = "Requires models/Qwen3-0.6B to be present"]
    fn test_load_qwen3_06b_f32() {
        let loader = SafeTensorsLoader::new("models/Qwen3-0.6B").unwrap();
        let weights = loader.load_all_weights::<f32>().unwrap();
        assert!(!weights.is_empty());
    }
}
