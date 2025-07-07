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
