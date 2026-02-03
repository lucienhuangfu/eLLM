#![feature(f16)]
use anyhow::{anyhow, Result};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde_json;
use std::collections::HashMap;
use std::env;
use std::f16;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

// 简化的配置结构，只包含我们需要的字段
#[derive(Debug, serde::Deserialize)]
struct SimpleConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
}

fn load_config<P: AsRef<Path>>(config_path: P) -> Result<SimpleConfig> {
    let file = File::open(config_path)?;
    let reader = BufReader::new(file);
    let config: SimpleConfig = serde_json::from_reader(reader)?;
    Ok(config)
}

fn load_weights_from_safetensors<P: AsRef<Path>>(
    model_path: P,
) -> Result<HashMap<String, Vec<f16>>> {
    let file = File::open(model_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let safetensors = SafeTensors::deserialize(&mmap)?;

    let mut weights = HashMap::new();

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

fn find_safetensors_file<P: AsRef<Path>>(model_dir: P) -> Result<std::path::PathBuf> {
    let model_dir = model_dir.as_ref();

    let patterns = [
        "model.safetensors",
        "pytorch_model.safetensors",
        "model-00001-of-00001.safetensors",
    ];

    for pattern in &patterns {
        let file_path = model_dir.join(pattern);
        if file_path.exists() {
            return Ok(file_path);
        }
    }

    let entries = std::fs::read_dir(model_dir)?;
    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        if file_name_str.starts_with("model-") && file_name_str.ends_with(".safetensors") {
            return Ok(entry.path());
        }
    }

    Err(anyhow!("No safetensors file found in the model directory"))
}

fn main() -> Result<()> {
    // 从命令行参数获取模型路径，或使用默认路径
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        // 默认路径，请根据你的实际路径修改
        "models/llama3-8b-instruct"
    };

    println!("Loading Llama3 8B model from: {}", model_path);

    // 构建配置文件和模型文件路径
    let model_dir = Path::new(model_path);
    let config_file = model_dir.join("config.json");

    if !config_file.exists() {
        eprintln!("❌ config.json not found in: {}", model_path);
        return Err(anyhow!("config.json not found"));
    }

    // 查找safetensors文件
    let model_file = match find_safetensors_file(model_dir) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("❌ Failed to find safetensors file: {}", e);
            return Err(e);
        }
    };

    println!("Found model file: {}", model_file.display());

    // 加载配置
    let config = match load_config(&config_file) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("❌ Failed to load config: {}", e);
            return Err(e);
        }
    };

    // 加载权重
    let weights = match load_weights_from_safetensors(&model_file) {
        Ok(weights) => weights,
        Err(e) => {
            eprintln!("❌ Failed to load weights: {}", e);
            return Err(e);
        }
    };

    println!("✅ Successfully loaded Llama3 8B model!");

    // 显示模型信息
    println!("\n📊 Model Configuration:");
    println!("  Model Type: {}", config.model_type);
    println!("  Hidden Size: {}", config.hidden_size);
    println!("  Layers: {}", config.num_hidden_layers);
    println!("  Attention Heads: {}", config.num_attention_heads);
    if let Some(kv_heads) = config.num_key_value_heads {
        println!("  Key-Value Heads: {}", kv_heads);
    }
    println!("  Vocabulary Size: {}", config.vocab_size);
    println!(
        "  Max Position Embeddings: {}",
        config.max_position_embeddings
    );
    println!("  RMS Norm Epsilon: {}", config.rms_norm_eps);

    // 计算模型参数和内存使用
    let mut total_params = 0;
    let mut total_memory_gb = 0.0;

    for (_, tensor) in &weights {
        total_params += tensor.len();
        // f16 每个参数占用2字节
        total_memory_gb += (tensor.len() * 2) as f64 / (1024.0 * 1024.0 * 1024.0);
    }

    println!("\n💾 Memory Usage:");
    println!("  Total Parameters: {:.2}B", total_params as f64 / 1e9);
    println!("  Memory Usage (f16): {:.2} GB", total_memory_gb);
    println!("  Loaded Tensors: {}", weights.len());

    // 验证关键层
    println!("\n🔍 Verifying Key Layers:");
    verify_model_layers(&config, &weights);

    println!("\n✅ Model loading and verification completed!");

    Ok(())
}

fn verify_model_layers(config: &SimpleConfig, weights: &HashMap<String, Vec<f16>>) {
    // 检查基础层
    let base_layers = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];

    for layer_name in &base_layers {
        if weights.contains_key(*layer_name) {
            let tensor = &weights[*layer_name];
            println!("  ✅ {}: {} params", layer_name, tensor.len());
        } else {
            println!("  ❌ Missing: {}", layer_name);
        }
    }

    // 检查transformer层
    let mut complete_layers = 0;
    for i in 0..config.num_hidden_layers {
        let layer_components = [
            format!("model.layers.{}.self_attn.q_proj.weight", i),
            format!("model.layers.{}.self_attn.k_proj.weight", i),
            format!("model.layers.{}.self_attn.v_proj.weight", i),
            format!("model.layers.{}.self_attn.o_proj.weight", i),
            format!("model.layers.{}.mlp.gate_proj.weight", i),
            format!("model.layers.{}.mlp.up_proj.weight", i),
            format!("model.layers.{}.mlp.down_proj.weight", i),
            format!("model.layers.{}.input_layernorm.weight", i),
            format!("model.layers.{}.post_attention_layernorm.weight", i),
        ];

        let mut layer_complete = true;
        for component in &layer_components {
            if !weights.contains_key(component) {
                layer_complete = false;
                break;
            }
        }

        if layer_complete {
            complete_layers += 1;
        } else {
            println!("  ❌ Layer {} incomplete", i);
        }
    }

    println!(
        "  ✅ Complete transformer layers: {}/{}",
        complete_layers, config.num_hidden_layers
    );

    // 显示一些大型张量的信息
    println!("\n📈 Large Tensors:");
    let mut large_tensors: Vec<_> = weights
        .iter()
        .filter(|(_, tensor)| tensor.len() > 1_000_000) // 大于1M参数
        .collect();
    large_tensors.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (name, tensor) in large_tensors.iter().take(5) {
        let size_mb = (tensor.len() * 2) as f64 / (1024.0 * 1024.0);
        println!(
            "  • {}: {:.1}M params ({:.1} MB)",
            name,
            tensor.len() as f64 / 1e6,
            size_mb
        );
    }
}
