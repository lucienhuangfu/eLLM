use minijinja::context;
use minijinja::Environment;
use std::error::Error;

pub struct ChatTemplate {
    _env: &'static Environment<'static>,
    template: minijinja::Template<'static, 'static>,
}

impl ChatTemplate {
    pub fn new(template_path: &str) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut env = Environment::new();
        let chat_template = std::fs::read_to_string(template_path)?;
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
    use minijinja::context;
    use serde_json::json;
    use std::error::Error;

    #[test]
    fn test_chat_template() -> Result<(), Box<dyn Error + Send + Sync>> {
        let template_path = "./models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";

        let messages = vec![
            ("system", "You are a helpful assistant."),
            ("user", "你好，世界！这是一次分词测试。"),
        ];

        let tester = ChatTemplate::new(template_path)?;
        let prompt = tester.apply_chat_template(&messages)?;
        println!("渲染后的 Prompt:\n{}", prompt);
        Ok(())
    }

    #[test]
    fn test_chat_template_multi_turn() -> Result<(), Box<dyn Error + Send + Sync>> {
        let template_path = "./models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";

        let messages = vec![
            ("system", "You are a helpful coding assistant."),
            ("user", "请帮我写一个 Rust 的快速排序函数。"),
            (
                "assistant",
                "当然可以。你希望是 in-place 版本，还是返回新数组的版本？",
            ),
            ("user", "in-place 版本，并加一个简单测试。"),
            (
                "assistant",
                "好的，我会给出一个泛型 in-place quicksort，并附带单元测试。",
            ),
        ];

        let tester = ChatTemplate::new(template_path)?;
        let prompt = tester.apply_chat_template(&messages)?;
        println!("多轮渲染后的 Prompt:\n{}", prompt);
        assert!(!prompt.trim().is_empty());
        Ok(())
    }

    #[test]
    fn test_chat_template_multi_turn_with_tools() -> Result<(), Box<dyn Error + Send + Sync>> {
        let template_path = "./models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";
        let tester = ChatTemplate::new(template_path)?;

        let tools = vec![
            json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string", "description": "City name" }
                        }
                    }
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get local time by timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": { "type": "string", "description": "IANA timezone" }
                        }
                    }
                }
            }),
        ];

        let messages = vec![
            json!({"role": "system", "content": "You are a helpful assistant."}),
            json!({"role": "user", "content": "帮我查下北京天气。"}),
            json!({
                "role": "assistant",
                "content": "我先查询一下北京天气。",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": { "city": "Beijing" }
                    }
                }]
            }),
            json!({"role": "tool", "content": "{\"city\":\"Beijing\",\"weather\":\"Sunny\",\"temp_c\":23}"}),
            json!({"role": "assistant", "content": "北京现在晴天，23°C。"}),
            json!({"role": "user", "content": "顺便告诉我北京时间。"}),
            json!({
                "role": "assistant",
                "content": "我来查一下北京时间。",
                "tool_calls": [{
                    "function": {
                        "name": "get_time",
                        "arguments": { "timezone": "Asia/Shanghai" }
                    }
                }]
            }),
            json!({"role": "tool", "content": "{\"timezone\":\"Asia/Shanghai\",\"time\":\"2026-02-24 16:30:00\"}"}),
            json!({"role": "assistant", "content": "北京时间是 2026-02-24 16:30:00。"}),
        ];

        let prompt = tester.template.render(context! {
            messages => messages,
            tools => tools,
            add_generation_prompt => false
        })?;

        println!("带工具多轮渲染后的 Prompt:\n{}", prompt);
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("<tool_response>"));
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("get_time"));
        Ok(())
    }
}
