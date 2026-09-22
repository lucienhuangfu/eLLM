use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// 规范聊天消息类型，定义于 runtime 层，此处 re-export 以保持 serving API 稳定。
pub use crate::runtime::loader::ChatMessage;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest<'a> {
    #[serde(borrow)]
    pub model: Cow<'a, str>,
    #[serde(borrow)]
    pub messages: Vec<ChatMessage<'a>>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    #[serde(borrow)]
    pub request_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    pub session_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    pub session_mode: Option<Cow<'a, str>>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse<'a> {
    pub id: Cow<'a, str>,
    pub object: Cow<'a, str>,
    pub created: u64,
    pub model: Cow<'a, str>,
    pub choices: Vec<ChatCompletionChoice<'a>>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice<'a> {
    pub index: u32,
    pub message: ChatMessage<'a>,
    pub finish_reason: Option<Cow<'a, str>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_serde_roundtrip() {
        let msg = ChatMessage::new("user", "hello");
        let json = serde_json::to_string(&msg).unwrap();
        let de: ChatMessage<'_> = serde_json::from_str(&json).unwrap();
        assert_eq!(de.role.as_ref(), "user");
        assert_eq!(de.content.as_ref(), "hello");
    }

    #[test]
    fn request_deserialize_with_defaults() {
        let json = r#"{"model":"gpt","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest<'_> = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_ref(), "gpt");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.stream, None);
        assert_eq!(req.temperature, None);
        assert_eq!(req.max_tokens, None);
    }
}
