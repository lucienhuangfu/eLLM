use anyhow::Result;
use std::path::Path;

use crate::llama::model_loader::{load_llama3_from_safetensors, SafeTensorsModelLoader};

/// Llama3 8B模型加载示例
pub fn load_llama3_8b_example() -> Result<()> {
    // 指定模型目录路径（需要根据实际路径修改）
    let model_dir = "path/to/your/llama3-8b-model";

    // 方法1：使用便民函数直接加载
    println!("Loading Llama3 8B model from safetensors...");
    let (config, weights) = load_llama3_from_safetensors(model_dir)?;

    println!("Model config loaded:");
    println!("  - Model type: {}", config.model_type);
    println!("  - Hidden size: {}", config.hidden_size);
    println!("  - Number of layers: {}", config.num_hidden_layers);
    println!(
        "  - Number of attention heads: {}",
        config.num_attention_heads
    );
    println!("  - Vocabulary size: {}", config.vocab_size);
    println!(
        "  - Max position embeddings: {}",
        config.max_position_embeddings
    );

    println!("Loaded {} weight tensors", weights.len());

    // 打印一些权重信息
    for (name, tensor) in weights.iter().take(5) {
        println!("  - {}: {} elements", name, tensor.len());
    }

    Ok(())
}

/// 使用SafeTensorsModelLoader的详细示例
pub fn detailed_loading_example() -> Result<()> {
    let model_dir = "path/to/your/llama3-8b-model";

    // 方法2：使用详细的加载器
    let loader = SafeTensorsModelLoader::new(model_dir)?;

    // 分别加载配置和权重
    let config = loader.load_config()?;
    let weights = loader.load_weights_f16()?;

    // 验证关键层的存在
    let expected_layers = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];

    for layer_name in &expected_layers {
        if weights.contains_key(*layer_name) {
            println!("✓ Found layer: {}", layer_name);
        } else {
            println!("✗ Missing layer: {}", layer_name);
        }
    }

    // 检查transformer层
    for i in 0..config.num_hidden_layers {
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

    Ok(())
}

/// 内存使用情况分析
pub fn analyze_memory_usage(model_dir: &str) -> Result<()> {
    let (config, weights) = load_llama3_from_safetensors(model_dir)?;

    let mut total_parameters = 0;
    let mut total_memory_mb = 0.0;

    for (name, tensor) in &weights {
        let param_count = tensor.len();
        total_parameters += param_count;

        // f16 占用2字节
        let memory_bytes = param_count * 2;
        let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);
        total_memory_mb += memory_mb;

        if name.contains("embed") || name.contains("lm_head") {
            println!(
                "Large tensor: {} - {:.2} MB ({} params)",
                name, memory_mb, param_count
            );
        }
    }

    println!("\nMemory Analysis:");
    println!("Total parameters: {:.2}B", total_parameters as f64 / 1e9);
    println!("Total memory (f16): {:.2} GB", total_memory_mb / 1024.0);
    println!("Expected for 8B model: ~16 GB");

    Ok(())
}
