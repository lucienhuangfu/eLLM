#![feature(f16)]

use ellm::auto::AutoTokenizer;
use std::f16;
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "checkpoints/Qwen3-0.6B".to_string());
    let tokenizer = AutoTokenizer::from_pretrained(&model_dir)?;

    let messages = [("user", "你好，请用一句话介绍 Rust。")];
    let prompt = tokenizer
        .apply_chat_template(&messages, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let token_ids = tokenizer.encode(&prompt);

    println!("Prompt: {}", prompt);
    println!("Token count: {}", token_ids.len());

    Ok(())
}
