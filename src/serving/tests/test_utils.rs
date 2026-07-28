use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use futures_util::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::num_traits::FromNumber;
use crate::operators::fake_echo::FakeEcho;
use crate::operators::operator::Operator;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::executor::executor_pool::ExecutorPool;
use crate::runtime::loader::{load_tiktoken, ChatTemplate};
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{BatchSequence, Phase, SessionMode, SlotManager, SlotState};
use crate::serving::server::build_router;
use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;

// ─── Tokenizers & Templates ────────────────────────────────────────────────

pub fn test_tokenizer() -> Arc<CoreBPE> {
    Arc::new(tiktoken_rs::r50k_base().unwrap_or_else(|_| {
        let mut vocab: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        let merges: FxHashMap<String, u32> = FxHashMap::default();
        for i in 0..100 {
            vocab.insert(format!("token_{}", i).into_bytes(), i as u32);
        }
        CoreBPE::new(vocab, merges, "bpe").unwrap()
    }))
}

pub fn test_chat_template() -> Arc<ChatTemplate> {
    Arc::new(
        ChatTemplate::from_template_source(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
                .to_string(),
        )
        .unwrap(),
    )
}

const QWEN3_MODEL_DIR: &str = "./models/Qwen3-Coder-30B-A3B-Instruct";

pub fn qwen3_tokenizer() -> Arc<CoreBPE> {
    let tokenizer_path = format!("{}/tokenizer.json", QWEN3_MODEL_DIR);
    let config_path = format!("{}/tokenizer_config.json", QWEN3_MODEL_DIR);
    Arc::new(load_tiktoken(&tokenizer_path, &config_path).unwrap())
}

pub fn qwen3_chat_template() -> Arc<ChatTemplate> {
    let template_path = format!("{}/chat_template.jinja", QWEN3_MODEL_DIR);
    Arc::new(ChatTemplate::new(&template_path).unwrap())
}

pub fn digit_tokens_r50k() -> Vec<usize> {
    let bpe = tiktoken_rs::r50k_base().unwrap();
    "0123456789"
        .chars()
        .map(|c| {
            let s: String = c.to_string();
            bpe.encode_with_special_tokens(&s)[0] as usize
        })
        .collect()
}

// ─── SlotManager builders ──────────────────────────────────────────────────

pub fn create_test_manager(
    batch_size: usize,
    timeout_ms: u64,
) -> (Arc<SlotManager<f16>>, Vec<usize>) {
    create_test_manager_with_mode(batch_size, timeout_ms, SessionMode::Reusable)
}

pub fn create_test_manager_with_mode(
    batch_size: usize,
    timeout_ms: u64,
    mode: SessionMode,
) -> (Arc<SlotManager<f16>>, Vec<usize>) {
    build_manager(batch_size, timeout_ms, mode, test_tokenizer(), test_chat_template(), true, true)
}

pub fn create_qwen3_test_manager_with_mode(
    batch_size: usize,
    timeout_ms: u64,
    mode: SessionMode,
) -> (Arc<SlotManager<f16>>, Vec<usize>) {
    create_qwen3_test_manager_with_parser(batch_size, timeout_ms, mode, true, true)
}

pub fn create_qwen3_test_manager_with_parser(
    batch_size: usize,
    timeout_ms: u64,
    mode: SessionMode,
    reasoning_parser_enabled: bool,
    tool_call_parser_enabled: bool,
) -> (Arc<SlotManager<f16>>, Vec<usize>) {
    build_manager(
        batch_size,
        timeout_ms,
        mode,
        qwen3_tokenizer(),
        qwen3_chat_template(),
        reasoning_parser_enabled,
        tool_call_parser_enabled,
    )
}

fn build_manager(
    batch_size: usize,
    timeout_ms: u64,
    mode: SessionMode,
    tokenizer: Arc<CoreBPE>,
    chat_template: Arc<ChatTemplate>,
    reasoning_parser_enabled: bool,
    tool_call_parser_enabled: bool,
) -> (Arc<SlotManager<f16>>, Vec<usize>) {
    let seq_len = 1024;
    let mut buffer = vec![0usize; batch_size * seq_len];
    let batch_sequences = Arc::new(SharedMut::new(BatchSequence::<f16> {
        sequences: buffer.as_mut_ptr(),
        batch_temperature: vec![<f16 as FromNumber>::from_f32(1.0); batch_size],
        row_size: batch_size,
        col_size: seq_len,
        tokenizer,
        chat_template,
    }));
    let batch_states = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::idle())
            .collect::<Vec<_>>(),
    ));
    let manager = Arc::new(SlotManager::new(
        batch_size,
        batch_sequences,
        batch_states,
        mode,
        timeout_ms,
        reasoning_parser_enabled,
        tool_call_parser_enabled,
    ));
    (manager, buffer)
}

// ─── Router builders ───────────────────────────────────────────────────────

pub fn create_test_router() -> (Router, Arc<SlotManager<f16>>, Vec<usize>) {
    let (manager, buffer) = create_test_manager(4, 1000);
    let router = build_router(Arc::clone(&manager));
    (router, manager, buffer)
}

pub fn create_test_router_with_mode(mode: SessionMode) -> (Router, Arc<SlotManager<f16>>, Vec<usize>) {
    let (manager, buffer) = create_test_manager_with_mode(4, 1000, mode);
    let router = build_router(Arc::clone(&manager));
    (router, manager, buffer)
}

pub fn create_qwen3_test_router_with_mode(
    mode: SessionMode,
) -> (Router, Arc<SlotManager<f16>>, Vec<usize>) {
    let (manager, buffer) = create_qwen3_test_manager_with_mode(4, 1000, mode);
    let router = build_router(Arc::clone(&manager));
    (router, manager, buffer)
}

// ─── Runtime / Scheduler helpers ───────────────────────────────────────────

pub fn start_runtime_with_fakeecho(
    manager: Arc<SlotManager<f16>>,
    eos_id: usize,
    thread_num: usize,
    tokens: Vec<usize>,
) -> Arc<Scheduler> {
    let (batch_size, seq_len, sequences_ptr) = manager.batch_sequences.with(|seq| {
        (seq.row_size, seq.col_size, seq.sequences)
    });

    let batch_states = manager.batch_states.clone();
    let scheduler = Arc::new(Scheduler::new(batch_size, seq_len, thread_num, batch_states));

    let fake_echo = FakeEcho::new(sequences_ptr, seq_len, eos_id, tokens);
    let operator_queue: Vec<Operator<f16>> = vec![Operator::FakeEcho(fake_echo)];

    let executor = ExecutorPool::new(operator_queue, Arc::clone(&scheduler), thread_num);
    executor.start();

    scheduler
}

// ─── Low-level slot helpers ────────────────────────────────────────────────

pub fn find_active_slot(manager: &SlotManager<f16>) -> Option<usize> {
    manager.batch_states.with(|slots| {
        slots.iter().position(|s| !matches!(s.phase, Phase::Start | Phase::Eos))
    })
}

pub fn start_generation_loop(manager: Arc<SlotManager<f16>>, generated_tokens: Vec<u32>) {
    tokio::spawn(async move {
        let slot_index = loop {
            if let Some(idx) = find_active_slot(&manager) {
                break idx;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        };

        manager.batch_states.with_mut(|slots| {
            let slot = &mut slots[slot_index];
            if slot.phase == Phase::Prefill {
                let prefill_end = slot.next_sequence_index + slot.filling_length();
                slot.next_sequence_index = prefill_end;
                slot.phase = Phase::Decode;
                slot.notify.notify_one();
            }
        });

        for &token_id in &generated_tokens {
            manager.batch_states.with_mut(|slots| {
                let slot = &mut slots[slot_index];
                let pos = slot.next_sequence_index;
                slot.next_sequence_index += 1;
                slot.sequence_length += 1;
                manager.batch_sequences.with_mut(|seq| {
                    let offset = slot_index * seq.col_size + pos;
                    unsafe {
                        *seq.sequences.add(offset) = token_id as usize;
                    }
                });
                slot.notify.notify_one();
            });
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        manager.batch_states.with_mut(|slots| {
            slots[slot_index].phase = Phase::Eos;
            slots[slot_index].notify.notify_one();
        });
    });
}

pub fn start_generation_worker(
    manager: Arc<SlotManager<f16>>,
    generated_tokens: Vec<u32>,
    total_requests: usize,
) {
    use std::collections::HashMap;
    let tokens_len = generated_tokens.len();

    tokio::spawn(async move {
        let mut slot_progress: HashMap<usize, usize> = HashMap::new();
        let mut completed = 0;

        loop {
            if completed >= total_requests {
                break;
            }

            let batch_size = manager.batch_states.with(|slots| slots.len());
            let mut any_activity = false;

            for slot_index in 0..batch_size {
                let (phase, _prompt_length, _next_idx) = manager.batch_states.with(|slots| {
                    let slot = &slots[slot_index];
                    (slot.phase, slot.prompt_length, slot.next_sequence_index)
                });

                match phase {
                    Phase::Prefill => {
                        manager.batch_states.with_mut(|slots| {
                            let slot = &mut slots[slot_index];
                            if slot.phase == Phase::Prefill {
                                let prefill_end =
                                    slot.next_sequence_index + slot.filling_length();
                                slot.next_sequence_index = prefill_end;
                                slot.phase = Phase::Decode;
                                slot.notify.notify_one();
                                any_activity = true;
                            }
                        });
                        slot_progress.insert(slot_index, 0);
                    }
                    Phase::Decode => {
                        let progress = slot_progress.get(&slot_index).copied().unwrap_or(0);
                        if progress < tokens_len {
                            let token_id = generated_tokens[progress];
                            manager.batch_states.with_mut(|slots| {
                                let slot = &mut slots[slot_index];
                                if slot.phase != Phase::Decode {
                                    return;
                                }
                                let pos = slot.next_sequence_index;
                                slot.next_sequence_index += 1;
                                slot.sequence_length += 1;
                                manager.batch_sequences.with_mut(|seq| {
                                    let offset = slot_index * seq.col_size + pos;
                                    unsafe {
                                        *seq.sequences.add(offset) = token_id as usize;
                                    }
                                });
                                slot.notify.notify_one();
                                any_activity = true;
                            });
                            slot_progress.insert(slot_index, progress + 1);
                        } else {
                            manager.batch_states.with_mut(|slots| {
                                let slot = &mut slots[slot_index];
                                if slot.phase == Phase::Decode {
                                    slot.phase = Phase::Eos;
                                    slot.notify.notify_one();
                                    any_activity = true;
                                    completed += 1;
                                }
                            });
                        }
                    }
                    _ => {}
                }
            }

            if !any_activity {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    });
}

// ─── HTTP / Body helpers ───────────────────────────────────────────────────

pub async fn collect_body(body: Body) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut stream = body.into_data_stream();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }
    bytes
}

pub fn parse_sse_events(body: &str) -> Vec<serde_json::Value> {
    let mut events = Vec::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                events.push(json);
            }
        }
    }
    events
}

// ─── High-level test helpers ───────────────────────────────────────────────

pub fn chat_request(body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(body).unwrap()))
        .unwrap()
}

pub fn simple_chat_body(user_content: &str, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": user_content}
        ],
        "stream": stream
    })
}

pub fn assert_sync_response_basic(json: &serde_json::Value) {
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "test-model");
    assert!(json["id"].is_string());
    assert!(json["created"].is_number());
    assert!(json["choices"].is_array());
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

pub fn assert_sync_content_ends_with(json: &serde_json::Value, eos_text: &str) -> String {
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be a string");
    assert!(!content.is_empty(), "generated content should not be empty");
    assert!(
        content.ends_with(eos_text),
        "content should end with eos token {}, got: {:?}",
        eos_text,
        content
    );
    content[..content.len() - eos_text.len()].to_string()
}

pub fn assert_digit_pattern(generated: &str) {
    assert!(!generated.is_empty(), "should have generated tokens before eos");
    let expected_pattern = "0123456789".repeat(generated.len() / 10 + 1);
    assert!(
        generated.chars().eq(expected_pattern.chars().take(generated.len())),
        "generated content should cycle through 0-9 digits, got: {:?}",
        generated
    );
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallInfo {
    pub name: String,
    pub arguments: String,
}

pub struct StreamResult {
    pub full_content: String,
    pub full_reasoning: String,
    pub tool_calls: Vec<ToolCallInfo>,
    pub has_role: bool,
    pub has_reasoning: bool,
    pub has_tool_calls: bool,
    pub finish_event_count: usize,
}

pub fn collect_stream_result(events: &[serde_json::Value]) -> StreamResult {
    let mut result = StreamResult {
        full_content: String::new(),
        full_reasoning: String::new(),
        tool_calls: Vec::new(),
        has_role: false,
        has_reasoning: false,
        has_tool_calls: false,
        finish_event_count: 0,
    };

    for event in events {
        let delta = &event["choices"][0]["delta"];

        if let Some(role) = delta["role"].as_str() {
            assert_eq!(role, "assistant");
            result.has_role = true;
        }

        if let Some(content) = delta["content"].as_str() {
            result.full_content.push_str(content);
        }

        if let Some(reasoning) = delta["reasoning_content"].as_str() {
            result.full_reasoning.push_str(reasoning);
            result.has_reasoning = true;
        }

        if let Some(tool_calls) = delta["tool_calls"].as_array() {
            result.has_tool_calls = true;
            for tc in tool_calls {
                let index = tc["index"].as_u64().unwrap_or(0) as usize;

                while result.tool_calls.len() <= index {
                    result.tool_calls.push(ToolCallInfo {
                        name: String::new(),
                        arguments: String::new(),
                    });
                }

                let has_name = tc["function"]["name"].is_string();
                if let Some(name) = tc["function"]["name"].as_str() {
                    result.tool_calls[index].name = name.to_string();
                }
                if let Some(args) = tc["function"]["arguments"].as_str() {
                    if has_name {
                        result.tool_calls[index].arguments = args.to_string();
                    }
                }
            }
        }

        if event["choices"][0]["finish_reason"] == "stop" {
            result.finish_event_count += 1;
        }
    }

    result
}

pub fn assert_sse_event_basics(event: &serde_json::Value) {
    assert_eq!(event["object"], "chat.completion.chunk");
    assert_eq!(event["model"], "test-model");
    assert!(event["id"].is_string());
    assert!(event["created"].is_number());
    assert!(event["choices"].is_array());
    assert_eq!(event["choices"][0]["index"], 0);
}

pub async fn send_with_retry(
    router: Router,
    body: &serde_json::Value,
    max_retries: usize,
    base_delay_ms: u64,
) -> (axum::response::Response, usize) {
    let mut retries = 0;
    loop {
        let request = chat_request(body);
        let response = router.clone().oneshot(request).await.unwrap();

        if response.status() == StatusCode::OK {
            return (response, retries);
        }

        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "unexpected status {}",
            response.status()
        );

        retries += 1;
        assert!(retries <= max_retries, "too many retries ({})", retries);

        let delay = base_delay_ms * (1 << retries.min(6));
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
}
