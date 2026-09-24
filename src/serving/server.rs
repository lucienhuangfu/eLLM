use async_stream::stream;
use axum::{
    body::{Body, Bytes},
    extract::State,
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use bytes::{BufMut, BytesMut};
use std::borrow::Cow;
use std::convert::Infallible;
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
    BadRequest(String),
    UnprocessableEntity(String),
    TokenizationError(String),
    SlotUnavailable(String),
    InternalError(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::BadRequest(msg) => write!(f, "Bad request: {}", msg),
            ApiError::UnprocessableEntity(msg) => write!(f, "Unprocessable entity: {}", msg),
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
            ApiError::BadRequest(msg) => {
                eprintln!("Bad request: {}", msg);
                (
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("Invalid request: {}", msg),
                )
            }
            ApiError::UnprocessableEntity(msg) => {
                eprintln!("Unprocessable entity: {}", msg);
                (
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                    format!("Unprocessable entity: {}", msg),
                )
            }
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

/// Writes complete SSE frames (`data: {...}\n\n`) directly into a caller-owned
/// [`BytesMut`], bypassing axum's `Event`/`Sse` re-serialization. The static JSON
/// envelope is pre-built once; per-chunk work is pure byte appends with no
/// intermediate `String` allocation.
struct SseWriter {
    /// Pre-built: `{"id":"...","object":"chat.completion.chunk","created":N,"model":"...","choices":[{"index":0,"delta":{`
    prefix: Bytes,
}

impl SseWriter {
    fn new(id: &str, created: u64, model: &str) -> Self {
        let mut prefix = BytesMut::with_capacity(192);
        prefix.put_slice(b"{\"id\":\"");
        push_json_escaped(&mut prefix, id);
        prefix.put_slice(b"\",\"object\":\"chat.completion.chunk\",\"created\":");
        prefix.put_slice(created.to_string().as_bytes());
        prefix.put_slice(b",\"model\":\"");
        push_json_escaped(&mut prefix, model);
        prefix.put_slice(b"\",\"choices\":[{\"index\":0,\"delta\":{");

        Self {
            prefix: prefix.freeze(),
        }
    }

    /// Write a content delta frame into `out`.
    fn write_content(&self, out: &mut BytesMut, role: bool, content: &str) {
        self.begin(out, role);
        out.put_slice(b"\"content\":\"");
        push_json_escaped(out, content);
        out.put_slice(b"\"},\"finish_reason\":null}]}\n\n");
    }

    /// Write a reasoning_content delta frame into `out`.
    fn write_reasoning(&self, out: &mut BytesMut, role: bool, reasoning: &str) {
        self.begin(out, role);
        out.put_slice(b"\"reasoning_content\":\"");
        push_json_escaped(out, reasoning);
        out.put_slice(b"\"},\"finish_reason\":null}]}\n\n");
    }

    /// Write a tool_call delta frame into `out`.
    fn write_tool_call_delta(
        &self,
        out: &mut BytesMut,
        role: bool,
        index: u32,
        name: Option<&str>,
        arguments: Option<&str>,
    ) {
        self.begin(out, role);
        out.put_slice(b"\"tool_calls\":[{\"index\":");
        out.put_slice(index.to_string().as_bytes());
        out.put_slice(b",\"type\":\"function\",\"function\":{");
        if let Some(n) = name {
            out.put_slice(b"\"name\":\"");
            push_json_escaped(out, n);
            out.put_u8(b'"');
            if arguments.is_some() {
                out.put_u8(b',');
            }
        }
        if let Some(args) = arguments {
            out.put_slice(b"\"arguments\":\"");
            push_json_escaped(out, args);
            out.put_u8(b'"');
        }
        out.put_slice(b"}}]},\"finish_reason\":null}]}\n\n");
    }

    /// Write the final finish frame (empty delta) into `out`.
    fn write_finish(&self, out: &mut BytesMut) {
        out.put_slice(b"data: ");
        out.put_slice(&self.prefix[..]);
        out.put_slice(b"},\"finish_reason\":\"stop\"}]}\n\n");
    }

    /// Emit the `data: ` prefix, the JSON envelope and the optional role field.
    #[inline]
    fn begin(&self, out: &mut BytesMut, role: bool) {
        out.put_slice(b"data: ");
        out.put_slice(&self.prefix[..]);
        if role {
            out.put_slice(b"\"role\":\"assistant\",");
        }
    }
}

/// Minimal JSON string escaping (RFC 8259), byte-oriented for [`BytesMut`].
/// Multi-byte UTF-8 sequences (bytes >= 0x80) are copied through untouched.
#[inline]
fn push_json_escaped(out: &mut BytesMut, s: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = s.as_bytes();
    let mut start = 0;
    for i in 0..bytes.len() {
        let b = bytes[i];
        let esc: &[u8] = match b {
            b'"' => b"\\\"",
            b'\\' => b"\\\\",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\t' => b"\\t",
            c if c < 0x20 => {
                out.put_slice(&bytes[start..i]);
                out.put_slice(b"\\u00");
                out.put_u8(HEX[(c >> 4) as usize]);
                out.put_u8(HEX[(c & 0x0f) as usize]);
                start = i + 1;
                continue;
            }
            _ => continue,
        };
        out.put_slice(&bytes[start..i]);
        out.put_slice(esc);
        start = i + 1;
    }
    out.put_slice(&bytes[start..]);
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

    /// Process one decoded token text, appending complete SSE frames to `out`.
    fn process_token(&mut self, text: &str, out: &mut BytesMut) {
        // Split borrows: parser (mut) and writer (shared) are independent fields.
        let parser = &mut self.parser;
        let writer = &self.writer;
        let role_sent = &mut self.role_sent;
        let tool_call_index = &mut self.tool_call_index;

        let events = parser.feed(text);
        for event in events {
            match event {
                ParserEvent::Content(content) => {
                    writer.write_content(out, !*role_sent, content);
                    *role_sent = true;
                }
                ParserEvent::Reasoning(reasoning) => {
                    writer.write_reasoning(out, !*role_sent, reasoning);
                    *role_sent = true;
                }
                ParserEvent::ToolCallDelta(fragment) => {
                    writer.write_tool_call_delta(
                        out,
                        !*role_sent,
                        *tool_call_index,
                        None,
                        Some(fragment),
                    );
                    *role_sent = true;
                }
                ParserEvent::ToolCall(tool_call) => {
                    let args_str = tool_call.arguments.to_string();
                    writer.write_tool_call_delta(
                        out,
                        !*role_sent,
                        *tool_call_index,
                        Some(&tool_call.name),
                        Some(&args_str),
                    );
                    *tool_call_index += 1;
                    *role_sent = true;
                }
            }
        }
    }

    /// Append the final finish frame to `out`.
    fn finish(&self, out: &mut BytesMut) {
        self.writer.write_finish(out);
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
    let max_slot_size = resolved_config
        .serve
        .as_ref()
        .and_then(|s| s.max_slot_size)
        .unwrap_or(batch_size);

    let ctx = initialize_runtime(
        resolved_config,
        api_server_count,
        batch_size,
        max_slot_size,
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
    body: Bytes,
) -> axum::response::Response {
    let request: ChatCompletionRequest<'_> = match serde_json::from_slice(&body) {
        Ok(request) => request,
        // 区分语法错误(400)与结构错误(422)，与 axum `Json` 提取器行为一致。
        Err(e) if e.is_data() => {
            return ApiError::UnprocessableEntity(e.to_string()).into_response()
        }
        Err(e) => return ApiError::BadRequest(e.to_string()).into_response(),
    };

    let request_id: Cow<'_, str> = request
        .request_id
        .unwrap_or_else(|| Cow::Owned(format!("chatcmpl-{}", uuid::Uuid::new_v4())));
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
            &request_id,
            &model,
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
            object: "chat.completion".into(),
            created,
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage::new("assistant", generated_text),
                finish_reason: Some("stop".into()),
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
    request_id: &str,
    model: &str,
    created: u64,
) -> axum::response::Response {
    let session_id = session_id.to_string();
    let parser_options = ParserOptions {
        rule: ParserRule::qwen(),
        reasoning_parser: slot_manager.reasoning_parser_enabled,
        tool_call_parser: slot_manager.tool_call_parser_enabled,
    };
    let parser = IncrementalStreamingParser::with_options(parser_options);
    let writer = SseWriter::new(request_id, created, model);
    let prompt_length = slot_manager.get_prompt_length(slot_index);
    let mut session = StreamSession::new(parser, writer, prompt_length);
    // Accumulates complete SSE frames for one wake-up; handed off as a single
    // zero-copy `Bytes` chunk (one allocation per notify batch, not per frame).
    let mut frame_buf = BytesMut::with_capacity(4096);

    let stream_body = stream! {
        loop {
            notifier.notified().await;

            let (token_index, phase) = slot_manager.get_token_index_and_phase(slot_index);
            let is_eos = matches!(phase, Phase::Eos);

            while session.last_emitted < token_index {
                let text = slot_manager.decode_single_token(slot_index, session.last_emitted);
                session.last_emitted += 1;
                session.process_token(&text, &mut frame_buf);
            }

            if is_eos {
                session.finish(&mut frame_buf);
            }

            if !frame_buf.is_empty() {
                // `mem::take` transfers the filled buffer's allocation to `Bytes`
                // with no copy; `frame_buf` becomes empty and refills next round.
                let chunk = std::mem::take(&mut frame_buf).freeze();
                yield Ok::<Bytes, Infallible>(chunk);
            }

            if is_eos {
                break;
            }
        }

        let sequence_length = slot_manager.get_next_sequence_index(slot_index);
        Arc::clone(&slot_manager).release_session(&session_id, sequence_length).await;
    };

    // Hand-built SSE response: `Body::from_stream` forwards the pre-framed bytes
    // verbatim, skipping axum's `Event`/`Sse` per-frame re-serialization.
    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
        ],
        Body::from_stream(stream_body),
    )
        .into_response()
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
        let w = SseWriter::new("chatcmpl-1", 123, "test-model");
        let mut out = BytesMut::new();
        w.write_content(&mut out, true, "hello");
        let frame = String::from_utf8(out.to_vec()).unwrap();
        assert!(frame.contains("\"role\":\"assistant\""));
        assert!(frame.contains("\"content\":\"hello\""));
        assert!(frame.contains("\"id\":\"chatcmpl-1\""));
        assert!(frame.contains("\"model\":\"test-model\""));
        // Full SSE framing: `data: ` prefix and `\n\n` terminator.
        assert!(frame.starts_with("data: {"));
        assert!(frame.ends_with("}\n\n"));
        // Payload must be valid JSON once the SSE framing is stripped.
        let json = frame
            .strip_prefix("data: ")
            .and_then(|s| s.strip_suffix("\n\n"))
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v["choices"][0]["delta"]["content"], "hello");
        assert_eq!(v["choices"][0]["finish_reason"], serde_json::Value::Null);
    }

    #[test]
    fn sse_writer_escapes_special_chars() {
        let w = SseWriter::new("id", 0, "m");
        let mut out = BytesMut::new();
        w.write_content(&mut out, false, "line1\nline2\"quote");
        let frame = String::from_utf8(out.to_vec()).unwrap();
        assert!(frame.contains("line1\\nline2\\\"quote"));
        // Escaping must not leak a raw newline that would break SSE framing.
        let json = frame
            .strip_prefix("data: ")
            .and_then(|s| s.strip_suffix("\n\n"))
            .unwrap();
        assert!(!json.contains('\n'));
    }

    #[test]
    fn sse_writer_finish() {
        let w = SseWriter::new("id", 0, "m");
        let mut out = BytesMut::new();
        w.write_finish(&mut out);
        let frame = String::from_utf8(out.to_vec()).unwrap();
        assert!(frame.contains("\"finish_reason\":\"stop\""));
        assert!(frame.contains("\"delta\":{}"));
        assert!(frame.starts_with("data: {"));
        assert!(frame.ends_with("}\n\n"));
    }
}
