use async_stream::stream;
use axum::{
    extract::State,
    response::sse::Event,
    response::{IntoResponse, Sse},
    routing::post,
    Json, Router,
};
use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::Notify;

use super::parser::{IncrementalStreamingParser, ParserEvent, ParserOptions, ParserRule};
use super::types::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
};
use crate::config::ResolvedConfig;
use crate::runtime::session::SessionMode;
use crate::runtime::session::{Phase, SlotManager};
use crate::runtime::{initialize_runtime, RuntimeContext};

// ─── Error Types ─────────────────────────────────────────────────────────────

/// Serving 模块的统一错误类型
#[derive(Debug)]
pub enum ApiError {
    TokenizationError(String),
    SlotUnavailable(String),
    InternalError(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::TokenizationError(msg) => write!(f, "Tokenization failed: {}", msg),
            ApiError::SlotUnavailable(msg) => write!(f, "Slot unavailable: {}", msg),
            ApiError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let is_slot_unavailable = matches!(self, ApiError::SlotUnavailable(_));
        let (status, message) = match self {
            ApiError::TokenizationError(msg) => {
                eprintln!("Tokenization error: {}", msg);
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Tokenization failed: {}", msg),
                )
            }
            ApiError::SlotUnavailable(msg) => {
                eprintln!("Slot unavailable: {}", msg);
                (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    format!("Service unavailable: {}", msg),
                )
            }
            ApiError::InternalError(msg) => {
                eprintln!("Internal error: {}", msg);
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Operation failed: {}", msg),
                )
            }
        };

        let mut response = (status, message).into_response();
        if is_slot_unavailable {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
        }
        response
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

// ─── SSE Writer (zero-alloc per-chunk serialization) ─────────────────────────

/// Pre-allocates the static JSON envelope and writes variable delta content
/// directly into a reusable buffer, eliminating per-chunk serde allocations.
/// Output is raw JSON (no SSE framing) — axum's `Event` handles `data:` prefix.
struct SseWriter {
    buf: String,
    /// Pre-built: `{"id":"...","object":"chat.completion.chunk","created":N,"model":"...","choices":[{"index":0,"delta":{`
    prefix: String,
}

impl SseWriter {
    fn new(id: &str, created: u64, model: &str) -> Self {
        let mut prefix = String::with_capacity(192);
        prefix.push_str("{\"id\":\"");
        push_json_escaped(&mut prefix, id);
        prefix.push_str("\",\"object\":\"chat.completion.chunk\",\"created\":");
        prefix.push_str(&created.to_string());
        prefix.push_str(",\"model\":\"");
        push_json_escaped(&mut prefix, model);
        prefix.push_str("\",\"choices\":[{\"index\":0,\"delta\":{");

        Self {
            buf: String::with_capacity(4096),
            prefix,
        }
    }

    /// Write a content delta chunk. Returns the complete SSE frame.
    fn write_content(&mut self, role: bool, content: &str) -> &str {
        self.buf.clear();
        self.buf.push_str(&self.prefix);
        if role {
            self.buf.push_str("\"role\":\"assistant\",");
        }
        self.buf.push_str("\"content\":\"");
        push_json_escaped(&mut self.buf, content);
        self.buf.push_str("\"},\"finish_reason\":null}]}");
        &self.buf
    }

    /// Write a reasoning_content delta chunk.
    fn write_reasoning(&mut self, role: bool, reasoning: &str) -> &str {
        self.buf.clear();
        self.buf.push_str(&self.prefix);
        if role {
            self.buf.push_str("\"role\":\"assistant\",");
        }
        self.buf.push_str("\"reasoning_content\":\"");
        push_json_escaped(&mut self.buf, reasoning);
        self.buf.push_str("\"},\"finish_reason\":null}]}");
        &self.buf
    }

    /// Write a tool_call delta chunk.
    fn write_tool_call_delta(
        &mut self,
        role: bool,
        index: u32,
        name: Option<&str>,
        arguments: Option<&str>,
    ) -> &str {
        self.buf.clear();
        self.buf.push_str(&self.prefix);
        if role {
            self.buf.push_str("\"role\":\"assistant\",");
        }
        self.buf.push_str("\"tool_calls\":[{\"index\":");
        self.buf.push_str(&index.to_string());
        self.buf.push_str(",\"type\":\"function\",\"function\":{");
        if let Some(n) = name {
            self.buf.push_str("\"name\":\"");
            push_json_escaped(&mut self.buf, n);
            self.buf.push('"');
            if arguments.is_some() {
                self.buf.push(',');
            }
        }
        if let Some(args) = arguments {
            self.buf.push_str("\"arguments\":\"");
            push_json_escaped(&mut self.buf, args);
            self.buf.push('"');
        }
        self.buf.push_str("}}]},\"finish_reason\":null}]}");
        &self.buf
    }

    /// Write the final finish chunk with empty delta.
    fn write_finish(&mut self) -> &str {
        self.buf.clear();
        self.buf.push_str(&self.prefix);
        self.buf.push_str("},\"finish_reason\":\"stop\"}]}");
        &self.buf
    }
}

/// Minimal JSON string escaping (RFC 8259).
#[inline]
fn push_json_escaped(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

// ─── Stream Session (state machine) ─────────────────────────────────────────

struct StreamSession {
    parser: IncrementalStreamingParser,
    writer: SseWriter,
    role_sent: bool,
    tool_call_index: u32,
    last_emitted: usize,
}

impl StreamSession {
    fn new(parser: IncrementalStreamingParser, writer: SseWriter, prompt_length: usize) -> Self {
        Self {
            parser,
            writer,
            role_sent: false,
            tool_call_index: 0,
            last_emitted: prompt_length,
        }
    }

    /// Process one decoded token text. Appends SSE frames to `out`.
    fn process_token(&mut self, text: &str, out: &mut String) {
        // Split borrows: parser and writer are independent fields.
        let parser = &mut self.parser;
        let writer = &mut self.writer;
        let role_sent = &mut self.role_sent;
        let tool_call_index = &mut self.tool_call_index;

        let events = parser.feed(text);
        for event in events {
            match event {
                ParserEvent::Content(content) => {
                    let frame = writer.write_content(!*role_sent, content);
                    *role_sent = true;
                    out.push_str(frame);
                    out.push('\n');
                }
                ParserEvent::Reasoning(reasoning) => {
                    let frame = writer.write_reasoning(!*role_sent, reasoning);
                    *role_sent = true;
                    out.push_str(frame);
                    out.push('\n');
                }
                ParserEvent::ToolCallDelta(fragment) => {
                    let frame = writer.write_tool_call_delta(
                        !*role_sent,
                        *tool_call_index,
                        None,
                        Some(fragment),
                    );
                    *role_sent = true;
                    out.push_str(frame);
                    out.push('\n');
                }
                ParserEvent::ToolCall(tool_call) => {
                    let args_str = tool_call.arguments.to_string();
                    let frame = writer.write_tool_call_delta(
                        !*role_sent,
                        *tool_call_index,
                        Some(&tool_call.name),
                        Some(&args_str),
                    );
                    *tool_call_index += 1;
                    *role_sent = true;
                    out.push_str(frame);
                    out.push('\n');
                }
            }
        }
    }

    /// Generate the final finish frame.
    fn finish(&mut self) -> &str {
        self.writer.write_finish()
    }
}

// ─── Server Entry Points ─────────────────────────────────────────────────────

pub async fn run(
    slot_manager: Arc<SlotManager<f16>>,
    host: &str,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", host, port);
    println!("启动事件驱动的 OpenAI 兼容服务器...");

    let app = build_router(slot_manager);
    let listener = TcpListener::bind(&addr).await?;

    println!("服务器运行在 http://{}", addr);
    println!("API 端点:");
    println!("  POST /v1/chat/completions - OpenAI 兼容的聊天完成");
    println!("  GET  /status - 服务器状态");

    axum::serve(listener, app).await?;
    Ok(())
}

pub fn initialize_serving_resources(
    resolved_config: &ResolvedConfig,
) -> Result<RuntimeContext<f16>, Box<dyn std::error::Error>> {
    let api_server_count = resolved_config
        .serve
        .as_ref()
        .map(|s| s.api_server_count)
        .unwrap_or(2);
    let batch_size = resolved_config.scheduler.max_num_seqs;
    let sequence_length = resolved_config
        .model
        .raw_config
        .max_model_len
        .unwrap_or(128);
    let chunk_size = resolved_config.scheduler.max_num_batched_tokens;
    let session_mode = if resolved_config.scheduler.dialogue_cache_enabled {
        SessionMode::Reusable
    } else {
        SessionMode::NonReusable
    };
    let slot_reuse_timeout_ms = resolved_config
        .serve
        .as_ref()
        .map(|s| s.slot_reuse_timeout_ms)
        .unwrap_or(30000);

    let ctx = initialize_runtime(
        resolved_config,
        api_server_count,
        batch_size,
        sequence_length,
        chunk_size,
        session_mode,
        slot_reuse_timeout_ms,
    )?;

    Ok(ctx)
}

// ─── Router ──────────────────────────────────────────────────────────────────

pub(crate) fn build_router(slot_manager: Arc<SlotManager<f16>>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route(
            "/status",
            axum::routing::get(|State(_): State<Arc<SlotManager<f16>>>| async {
                Json(serde_json::json!({
                    "status": "running",
                    "mode": "inlined_scheduler",
                    "info": "Scheduler is inlined in worker loop, executed by leader thread"
                }))
            }),
        )
        .with_state(slot_manager)
}

// ─── Request Handler ─────────────────────────────────────────────────────────

async fn chat_completions(
    State(slot_manager): State<Arc<SlotManager<f16>>>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let request_id = request
        .request_id
        .unwrap_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4()));
    let is_stream = request.stream.unwrap_or(false);
    let model = request.model;

    let session_id = request.session_id.unwrap_or_else(|| request_id.clone());

    let handle = match slot_manager.acquire_session(&session_id).await {
        Ok(h) => h,
        Err(e) => return e.into_response(),
    };

    let slot_index = handle.slot_index;

    let (_write_len, notifier) = match slot_manager
        .write_prompts(
            slot_index,
            &session_id,
            &request.messages,
            request.temperature,
        )
        .await
    {
        Ok(result) => result,
        Err(e) => {
            Arc::clone(&slot_manager)
                .release_session(&session_id, 0)
                .await;
            return e.into_response();
        }
    };

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        build_stream_response(
            Arc::clone(&slot_manager),
            slot_index,
            &session_id,
            notifier,
            request_id,
            model,
            created,
        )
    } else {
        while !slot_manager.is_eos(slot_index) {
            notifier.notified().await;
        }

        let generated_text = slot_manager.decode_generated_text(slot_index);
        let sequence_length = slot_manager.get_next_sequence_index(slot_index);
        Arc::clone(&slot_manager)
            .release_session(&session_id, sequence_length)
            .await;

        Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: generated_text,
                },
                finish_reason: Some("stop".to_string()),
            }],
        })
        .into_response()
    }
}

// ─── Stream Response Builder ─────────────────────────────────────────────────

fn build_stream_response(
    slot_manager: Arc<SlotManager<f16>>,
    slot_index: usize,
    session_id: &str,
    notifier: Arc<Notify>,
    request_id: String,
    model: String,
    created: u64,
) -> axum::response::Response {
    let session_id = session_id.to_string();
    let parser_options = ParserOptions {
        rule: ParserRule::qwen(),
        reasoning_parser: slot_manager.reasoning_parser_enabled,
        tool_call_parser: slot_manager.tool_call_parser_enabled,
    };
    let parser = IncrementalStreamingParser::with_options(parser_options);
    let writer = SseWriter::new(&request_id, created, &model);
    let prompt_length = slot_manager.get_prompt_length(slot_index);
    let mut session = StreamSession::new(parser, writer, prompt_length);
    let mut frame_buf = String::with_capacity(4096);

    let stream_body = stream! {
        loop {
            notifier.notified().await;

            let (token_index, phase) = slot_manager.get_token_index_and_phase(slot_index);
            let is_eos = matches!(phase, Phase::Eos);

            while session.last_emitted < token_index {
                let text = slot_manager.decode_single_token(slot_index, session.last_emitted);
                session.last_emitted += 1;

                frame_buf.clear();
                session.process_token(&text, &mut frame_buf);

                if !frame_buf.is_empty() {
                    for json_str in frame_buf.split('\n').filter(|s| !s.is_empty()) {
                        yield Ok::<Event, axum::Error>(Event::default().data(json_str.to_string()));
                    }
                }
            }

            if is_eos {
                let finish_frame = session.finish().to_string();
                yield Ok::<Event, axum::Error>(Event::default().data(finish_frame));
                break;
            }
        }

        let sequence_length = slot_manager.get_next_sequence_index(slot_index);
        Arc::clone(&slot_manager).release_session(&session_id, sequence_length).await;
    };

    Sse::new(stream_body).into_response()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    #[test]
    fn error_display_messages() {
        assert_eq!(
            format!("{}", ApiError::TokenizationError("oops".into())),
            "Tokenization failed: oops"
        );
        assert_eq!(
            format!("{}", ApiError::SlotUnavailable("busy".into())),
            "Slot unavailable: busy"
        );
        assert_eq!(
            format!("{}", ApiError::InternalError("fail".into())),
            "Internal error: fail"
        );
    }

    #[test]
    fn tokenization_error_status_500() {
        let err = ApiError::TokenizationError("bad token".into());
        let resp = err.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn sse_writer_content_basic() {
        let mut w = SseWriter::new("chatcmpl-1", 123, "test-model");
        let frame = w.write_content(true, "hello").to_string();
        assert!(frame.contains("\"role\":\"assistant\""));
        assert!(frame.contains("\"content\":\"hello\""));
        assert!(frame.contains("\"id\":\"chatcmpl-1\""));
        assert!(frame.contains("\"model\":\"test-model\""));
        assert!(frame.starts_with('{'));
        assert!(frame.ends_with('}'));
        // Must be valid JSON
        let v: serde_json::Value = serde_json::from_str(&frame).unwrap();
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v["choices"][0]["delta"]["content"], "hello");
        assert_eq!(v["choices"][0]["finish_reason"], serde_json::Value::Null);
    }

    #[test]
    fn sse_writer_escapes_special_chars() {
        let mut w = SseWriter::new("id", 0, "m");
        let frame = w.write_content(false, "line1\nline2\"quote").to_string();
        assert!(frame.contains("line1\\nline2\\\"quote"));
    }

    #[test]
    fn sse_writer_finish() {
        let mut w = SseWriter::new("id", 0, "m");
        let frame = w.write_finish().to_string();
        assert!(frame.contains("\"finish_reason\":\"stop\""));
        assert!(frame.contains("\"delta\":{}"));
    }
}
