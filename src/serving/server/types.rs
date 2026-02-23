use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionRequest {
    pub(super) model: String,
    pub(super) messages: Vec<ChatMessage>,
    pub(super) stream: Option<bool>,
    pub(super) temperature: Option<f32>,
    pub(super) max_tokens: Option<usize>,
    pub(super) top_p: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(super) struct ChatMessage {
    pub(super) role: String,
    pub(super) content: String,
}

#[derive(Debug, Serialize)]
pub(super) struct ChatCompletionResponse {
    pub(super) id: String,
    pub(super) object: String,
    pub(super) created: u64,
    pub(super) model: String,
    pub(super) choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Serialize)]
pub(super) struct ChatCompletionChoice {
    pub(super) index: u32,
    pub(super) message: ChatMessage,
    pub(super) finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct StreamResponse {
    pub(super) id: String,
    pub(super) object: String,
    pub(super) created: u64,
    pub(super) model: String,
    pub(super) choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub(super) struct StreamChoice {
    pub(super) index: u32,
    pub(super) delta: ChatMessage,
    pub(super) finish_reason: Option<String>,
}
