use std::collections::HashMap;
use std::f16;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Result};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};

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

    pub fn load_all_weights_f16(&self) -> Result<HashMap<String, Vec<f16>>> {
        let mut all_weights = HashMap::with_capacity(512);

        for model_file in &self.model_files {
            let file = File::open(model_file)?;
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
                    Dtype::BF16 => tensor_view
                        .data()
                        .chunks_exact(2)
                        .map(|chunk| {
                            let val_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                            let val_f32 = f32::from_bits((val_u16 as u32) << 16);
                            val_f32 as f16
                        })
                        .collect(),
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

    pub fn merge_moe(&self, weights: &mut HashMap<String, Vec<f16>>) -> Result<()> {
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
    fn test_merge_moe_qwen() {
        let loader = SafeTensorsLoader {
            model_files: vec![],
        };

        let mut weights = HashMap::new();
        let prefix = "model.layers.0.mlp.";

        weights.insert(
            format!("{}experts.0.down_proj.weight", prefix),
            vec![1.0f16],
        );
        weights.insert(
            format!("{}experts.0.gate_proj.weight", prefix),
            vec![2.0f16],
        );
        weights.insert(format!("{}experts.0.up_proj.weight", prefix), vec![3.0f16]);

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

        let down_key = format!("{}experts.down_proj.weight", prefix);
        assert!(weights.contains_key(&down_key));
        let down_merged = weights.get(&down_key).unwrap();
        assert_eq!(down_merged.len(), 2);
        assert_eq!(down_merged[0], 1.0f16);
        assert_eq!(down_merged[1], 4.0f16);

        let gate_key = format!("{}experts.gate_proj.weight", prefix);
        assert!(weights.contains_key(&gate_key));
        let gate_merged = weights.get(&gate_key).unwrap();
        assert_eq!(gate_merged.len(), 2);
        assert_eq!(gate_merged[0], 2.0f16);
        assert_eq!(gate_merged[1], 5.0f16);

        let up_key = format!("{}experts.up_proj.weight", prefix);
        assert!(weights.contains_key(&up_key));
        let up_merged = weights.get(&up_key).unwrap();
        assert_eq!(up_merged.len(), 2);
        assert_eq!(up_merged[0], 3.0f16);
        assert_eq!(up_merged[1], 6.0f16);
    }
}
