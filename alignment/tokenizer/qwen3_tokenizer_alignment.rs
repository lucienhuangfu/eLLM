use ellm::runtime::chat_template::ChatTemplate;
use ellm::runtime::tokenizer_loader::load_tiktoken;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-0.6B".to_string());
    let tokenizer_path = format!("{model_dir}/tokenizer.json");
    let tokenizer_config_path = format!("{model_dir}/tokenizer_config.json");
    let chat_template_path = format!("{model_dir}/chat_template.jinja");

    let messages = [("user", "你好，请用一句话介绍 Rust。")];
    let chat_template = ChatTemplate::from_model_files(&chat_template_path, &tokenizer_config_path)?;
    let prompt = chat_template.apply_chat_template(&messages, true)?;
    let tokenizer = load_tiktoken(&tokenizer_path, &tokenizer_config_path)?;
    let token_ids = tokenizer.encode_with_special_tokens(&prompt);

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model_dir": model_dir,
            "messages": messages
                .iter()
                .map(|(role, content)| json!({"role": role, "content": content}))
                .collect::<Vec<_>>(),
            "add_generation_prompt": true,
            "rendered_prompt": prompt,
            "token_ids": token_ids,
        }))?
    );

    Ok(())
}
