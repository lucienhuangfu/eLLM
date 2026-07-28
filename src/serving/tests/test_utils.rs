use axum::body::Body;
use axum::Router;
use futures_util::StreamExt;
use std::sync::Arc;

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
    let seq_len = 1024;
    let mut buffer = vec![0usize; batch_size * seq_len];
    let batch_sequences = Arc::new(SharedMut::new(BatchSequence::<f16> {
        sequences: buffer.as_mut_ptr(),
        batch_temperature: vec![<f16 as FromNumber>::from_f32(1.0); batch_size],
        row_size: batch_size,
        col_size: seq_len,
        tokenizer: test_tokenizer(),
        chat_template: test_chat_template(),
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
        true,
        true,
    ));
    (manager, buffer)
}

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
    let seq_len = 1024;
    let mut buffer = vec![0usize; batch_size * seq_len];
    let batch_sequences = Arc::new(SharedMut::new(BatchSequence::<f16> {
        sequences: buffer.as_mut_ptr(),
        batch_temperature: vec![<f16 as FromNumber>::from_f32(1.0); batch_size],
        row_size: batch_size,
        col_size: seq_len,
        tokenizer: qwen3_tokenizer(),
        chat_template: qwen3_chat_template(),
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

pub fn create_qwen3_test_router_with_mode(
    mode: SessionMode,
) -> (Router, Arc<SlotManager<f16>>, Vec<usize>) {
    let (manager, buffer) = create_qwen3_test_manager_with_mode(4, 1000, mode);
    let router = build_router(Arc::clone(&manager));
    (router, manager, buffer)
}

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
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
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
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
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
                let (phase, prompt_length, next_idx) = manager.batch_states.with(|slots| {
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
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        }
    });
}

pub async fn collect_body(body: Body) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut stream = body.into_data_stream();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }
    bytes
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

pub fn start_runtime_with_fakeecho(
    manager: Arc<SlotManager<f16>>,
    eos_id: usize,
    thread_num: usize,
    tokens: Vec<usize>,
    max_gen_tokens: usize,
) -> Arc<Scheduler> {
    let (batch_size, seq_len, sequences_ptr) = manager.batch_sequences.with(|seq| {
        (seq.row_size, seq.col_size, seq.sequences)
    });

    let batch_states = manager.batch_states.clone();
    let scheduler = Arc::new(Scheduler::new(batch_size, seq_len, thread_num, batch_states));

    let fake_echo = FakeEcho::new(sequences_ptr, seq_len, eos_id, tokens, max_gen_tokens);
    let operator_queue: Vec<Operator<f16>> = vec![Operator::FakeEcho(fake_echo)];

    let executor = ExecutorPool::new(operator_queue, Arc::clone(&scheduler), thread_num);
    executor.start();

    scheduler
}
