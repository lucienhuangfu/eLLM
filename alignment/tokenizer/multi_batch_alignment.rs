#![feature(f16)]

use ellm::auto::AutoTokenizer;

const DEFAULT_PROMPTS: &[&str] = &[
    "你好，请用一句话介绍 Rust。",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "请解释什么是机器学习。",
];

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "checkpoints/Qwen3-0.6B".to_string());
    let tokenizer = AutoTokenizer::from_pretrained(&model_dir)?;

    for &prompt in DEFAULT_PROMPTS {
        let messages = [("user", prompt)];
        let template_prompt = tokenizer
            .apply_chat_template(&messages, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let token_ids = tokenizer.encode(&template_prompt);
        println!("Prompt: {}, Token count: {}", prompt, token_ids.len());
    }

    Ok(())
}
