use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Result};
use memmap2::MmapOptions;
use safetensors::SafeTensors;

use super::from_safetensors::FromSafetensors;

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

    pub fn load_all_weights<T: FromSafetensors>(&self) -> Result<HashMap<String, Vec<T>>> {
        let mut all_weights = HashMap::with_capacity(512);

        for model_file in &self.model_files {
            let file = File::open(model_file)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let safetensors = SafeTensors::deserialize(&mmap)?;

            for (name, tensor_view) in safetensors.tensors() {
                let data = T::from_tensor_view(&tensor_view)?;
                all_weights.insert(name.to_string(), data);
            }
        }

        Ok(all_weights)
    }

    pub fn load_all_weights_f16(&self) -> Result<HashMap<String, Vec<f16>>> {
        self.load_all_weights::<f16>()
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
