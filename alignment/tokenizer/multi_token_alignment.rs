#![feature(f16)]

use ellm::runtime::loader::load_tiktoken;
use ellm::runtime::loader::ChatTemplate;

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-0.6B".to_string());
    let tokenizer_path = format!("{model_dir}/tokenizer.json");
    let tokenizer_config_path = format!("{model_dir}/tokenizer_config.json");
    let chat_template_path = format!("{model_dir}/chat_template.jinja");

    let chat_template = ChatTemplate::from_model_files(&chat_template_path, &tokenizer_config_path)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let tokenizer =
        load_tiktoken(&tokenizer_path, &tokenizer_config_path).map_err(|e| anyhow::anyhow!(e))?;

    let messages = [("user", "你好，请用一句话介绍 Rust。")];
    let prompt = chat_template
        .apply_chat_template(&messages, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let token_ids = tokenizer.encode_with_special_tokens(&prompt);

    println!("Prompt: {}", prompt);
    println!("Token count: {}", token_ids.len());

    Ok(())
}
