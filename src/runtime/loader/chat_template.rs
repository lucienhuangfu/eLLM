use std::borrow::Cow;
use std::fmt;
use std::path::Path;

use anyhow::Result;
use minijinja::context;
use minijinja::value::{from_args, Value};
use minijinja::Environment;
use minijinja::{Error as MiniJinjaError, ErrorKind, State};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct TokenizerConfigTemplate {
    chat_template: String,
}

/// 规范聊天消息类型，全链路唯一表示。
/// 请求侧零拷贝借用 body（`Cow::Borrowed`），响应侧持有生成文本（`Cow::Owned`）；
/// 含转义符的字符串由 serde_json 自动回退 `Owned`。直接由 minijinja 序列化，
/// 避免经 `serde_json::Value` 中转而拷贝 content。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage<'a> {
    #[serde(borrow)]
    pub role: Cow<'a, str>,
    #[serde(borrow)]
    pub content: Cow<'a, str>,
}

impl<'a> ChatMessage<'a> {
    /// 便捷构造：`&str` 借用为 `Cow::Borrowed`，`String` 转为 `Cow::Owned`。
    pub fn new<R: Into<Cow<'a, str>>, C: Into<Cow<'a, str>>>(role: R, content: C) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

pub struct ChatTemplate {
    _env: &'static Environment<'static>,
    template: minijinja::Template<'static, 'static>,
}

impl ChatTemplate {
    pub fn new(template_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let chat_template = std::fs::read_to_string(template_path)?;
        Self::from_template_source(chat_template)
    }

    pub fn from_tokenizer_config(
        tokenizer_config_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let content = std::fs::read_to_string(tokenizer_config_path)?;
        let config: TokenizerConfigTemplate = serde_json::from_str(&content)?;
        Self::from_template_source(config.chat_template)
    }

    pub fn from_model_files(
        chat_template_path: &str,
        tokenizer_config_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if Path::new(chat_template_path).exists() {
            return Self::new(chat_template_path);
        }

        Self::from_tokenizer_config(tokenizer_config_path)
    }

    pub fn from_template_source(
        chat_template: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(
            |_state: &State, value: &Value, method: &str, args: &[Value]| {
                let Some(input) = value.as_str() else {
                    return Err(MiniJinjaError::from(ErrorKind::UnknownMethod));
                };
                let (needle,): (String,) = from_args(args)?;

                match method {
                    "startswith" => Ok(Value::from(input.starts_with(&needle))),
                    "endswith" => Ok(Value::from(input.ends_with(&needle))),
                    _ => Err(MiniJinjaError::from(ErrorKind::UnknownMethod)),
                }
            },
        );
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
        messages: &[ChatMessage<'_>],
        add_generation_prompt: bool,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let prompt = self.template.render(context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt
        })?;

        Ok(prompt)
    }
}

impl fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatTemplate").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPLATE_PATH: &str = "./models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";

    #[test]
    fn test_chat_template() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let messages = vec![
            ChatMessage::new("system", "You are a helpful assistant."),
            ChatMessage::new("user", "你好，世界！这是一次分词测试。"),
        ];

        let tester = ChatTemplate::new(TEMPLATE_PATH)?;
        let prompt = tester.apply_chat_template(&messages, false)?;
        println!("渲染后的 Prompt:\n{}", prompt);
        Ok(())
    }

    #[test]
    fn test_chat_template_multi_turn() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let messages = vec![
            ChatMessage::new("system", "You are a helpful coding assistant."),
            ChatMessage::new("user", "请帮我写一个 Rust 的快速排序函数。"),
            ChatMessage::new(
                "assistant",
                "当然可以。你希望是 in-place 版本，还是返回新数组的版本？",
            ),
            ChatMessage::new("user", "in-place 版本，并加一个简单测试。"),
            ChatMessage::new(
                "assistant",
                "好的，我会给出一个泛型 in-place quicksort，并附带单元测试。",
            ),
        ];

        let tester = ChatTemplate::new(TEMPLATE_PATH)?;
        let prompt = tester.apply_chat_template(&messages, false)?;
        println!("多轮渲染后的 Prompt:\n{}", prompt);
        assert!(!prompt.trim().is_empty());
        Ok(())
    }

    #[test]
    fn test_chat_template_multi_turn_with_tools(
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let tester = ChatTemplate::new(TEMPLATE_PATH)?;

        let tools = vec![
            serde_json::json!({
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
            serde_json::json!({
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
            serde_json::json!({"role": "system", "content": "You are a helpful assistant."}),
            serde_json::json!({"role": "user", "content": "帮我查下北京天气。"}),
            serde_json::json!({
                "role": "assistant",
                "content": "我先查询一下北京天气。",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": { "city": "Beijing" }
                    }
                }]
            }),
            serde_json::json!({"role": "tool", "content": "{\"city\":\"Beijing\",\"weather\":\"Sunny\",\"temp_c\":23}"}),
            serde_json::json!({"role": "assistant", "content": "北京现在晴天，23°C。"}),
            serde_json::json!({"role": "user", "content": "顺便告诉我北京时间。"}),
            serde_json::json!({
                "role": "assistant",
                "content": "我来查一下北京时间。",
                "tool_calls": [{
                    "function": {
                        "name": "get_time",
                        "arguments": { "timezone": "Asia/Shanghai" }
                    }
                }]
            }),
            serde_json::json!({"role": "tool", "content": "{\"timezone\":\"Asia/Shanghai\",\"time\":\"2026-02-24 16:30:00\"}"}),
            serde_json::json!({"role": "assistant", "content": "北京时间是 2026-02-24 16:30:00。"}),
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
