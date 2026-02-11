use minijinja::context;
use minijinja::Environment;
use std::error::Error;
struct ChatTemplate {
    env: &'static Environment<'static>,
    template: minijinja::Template<'static>,
}

impl ChatTemplate {
    fn new(template_path: &str) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut env = Environment::new();
        let chat_template = std::fs::read_to_string(template_path)?;
        let template_name = "chat";

        env.add_template(template_name, &chat_template)?;
        let env = Box::leak(Box::new(env));
        let template = env.get_template(template_name)?;

        Ok(Self { env, template })
    }

    fn apply_chat_template(
        &self,
        messages: &[(&str, &str)],
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let prompt = self.template.render(context! {
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

        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::ChatTemplate;
    use std::error::Error;

    #[test]
    fn test_chat_template() -> Result<(), Box<dyn Error + Send + Sync>> {
        let template_path = "./chat_template.jinja";

        // 模拟一个对话
        let messages = vec![
            ("system", "You are a helpful assistant."),
            ("user", "你好，世界！这是一次分词测试。"),
        ];

        let tester = ChatTemplate::new(template_path)?;
        let prompt = tester.apply_chat_template(&messages)?;
        println!("渲染后的 Prompt:\n{}", prompt);
        Ok(())
    }
}
