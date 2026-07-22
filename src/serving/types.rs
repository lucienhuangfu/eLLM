use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub request_id: Option<String>,
    pub session_id: Option<String>,
    pub session_mode: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: StreamToolFunction,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamToolFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_serde_roundtrip() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let de: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(de.role, "user");
        assert_eq!(de.content, "hello");
    }

    #[test]
    fn request_deserialize_with_defaults() {
        let json = r#"{"model":"gpt","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gpt");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.stream, None);
        assert_eq!(req.temperature, None);
        assert_eq!(req.max_tokens, None);
    }

    #[test]
    fn stream_delta_skip_none() {
        let delta = StreamDelta::default();
        let json = serde_json::to_string(&delta).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn stream_delta_with_role_and_content() {
        let delta = StreamDelta {
            role: Some("assistant".to_string()),
            content: Some("hi".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&delta).unwrap();
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(val["role"], "assistant");
        assert_eq!(val["content"], "hi");
        assert!(val.get("reasoning_content").is_none());
    }
}
