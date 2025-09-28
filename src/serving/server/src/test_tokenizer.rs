use std::error::Error;
use std::sync::Arc;
use tokenizers::Tokenizer;
use minijinja::Environment;
use minijinja::context;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let tokenizer = Arc::new(Tokenizer::from_file("./tokenizer.json")?);
    let chat_template = std::fs::read_to_string("./chat_template.jinja")?;

    // 模拟一个对话
    let messages = vec![
        ("system", "You are a helpful assistant."),
        ("user", "你好，世界！这是一次分词测试。"),
    ];

    // 用 chat_template 渲染
    let mut env = Environment::new();
    let tmpl = env.template_from_str(&chat_template)?;

    // 用 minijinja 的 context 宏传递 messages
    let rendered = tmpl.render(context! {
        messages => messages
            .iter()
            .map(|(role, content)| {
                serde_json::json!({
                    "role": role,
                    "content": content
                })
            })
            .collect::<Vec<_>>()
    })?;

    println!("渲染后的 Prompt:\n{}", rendered);

    // 分词
    let encoding = tokenizer.encode(rendered.clone(), true)?;
    println!("Token IDs: {:?}", encoding.get_ids());
    println!("Decode: {}", tokenizer.decode(encoding.get_ids(), true)?);

    Ok(())
}
