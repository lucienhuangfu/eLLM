use std::fs;

use minijinja::context;
use minijinja::Environment;

pub struct ChatTemplate {
    _env: &'static Environment<'static>,
    template: minijinja::Template<'static, 'static>,
}

impl ChatTemplate {
    pub fn new(template_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut env = Environment::new();
        let chat_template = fs::read_to_string(template_path)?;
        let chat_template = Box::leak(chat_template.into_boxed_str());
        let template_name = "chat";

        env.add_template(template_name, chat_template)?;
        let env = Box::leak(Box::new(env));
        let template = env.get_template(template_name)?;

        Ok(Self {
            _env: env,
            template,
        })
    }

    pub fn apply_chat_template(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let prompt = self.template.render(context! {
            messages => messages
                .iter()
                .map(|(role, content)| {
                    serde_json::json!({
                        "role": role,
                        "content": content
                    })
                })
                .collect::<Vec<_>>(),
            add_generation_prompt => add_generation_prompt
        })?;

        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::ChatTemplate;

    const QWEN3_TEMPLATE_PATH: &str = "./models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";

    #[test]
    fn test_chat_template() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let template = ChatTemplate::new(QWEN3_TEMPLATE_PATH)?;
        let messages = vec![
            ("system", "You are a helpful assistant."),
            ("user", "你好，世界！这是一次分词测试。"),
        ];

        let prompt = template.apply_chat_template(&messages, false)?;
        println!("渲染后的 Prompt:\n{}", prompt);
        Ok(())
    }
}
