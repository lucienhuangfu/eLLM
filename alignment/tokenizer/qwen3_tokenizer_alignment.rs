use ellm::auto::AutoTokenizer;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "checkpoints/Qwen3-0.6B".to_string());
    let messages = [("user", "你好，请用一句话介绍 Rust。")];
    let tokenizer = AutoTokenizer::from_pretrained(&model_dir)?;
    let prompt = tokenizer.apply_chat_template(&messages, true)?;
    let token_ids = tokenizer.encode(&prompt);

    println!(
        "Prompt: {}\nToken IDs: {:?}\nToken count: {}",
        prompt,
        token_ids,
        token_ids.len()
    );

    let text = tokenizer.decode(&token_ids).unwrap();
    println!("Decoded text: {}", text);

    Ok(())
}
