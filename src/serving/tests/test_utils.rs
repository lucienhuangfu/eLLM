use axum::body::Body;
use axum::Router;
use futures_util::StreamExt;
use std::sync::Arc;

use crate::num_traits::FromNumber;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::loader::ChatTemplate;
use crate::runtime::session::{BatchSequence, Phase, SessionMode, SlotManager, SlotState};
use crate::serving::server::build_router;
use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;

pub fn test_tokenizer() -> Arc<CoreBPE> {
    Arc::new(
        crate::runtime::loader::load_tiktoken("gpt2", "gpt2").unwrap_or_else(|_| {
            let mut vocab: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
            let merges: FxHashMap<String, u32> = FxHashMap::default();
            for i in 0..100 {
                vocab.insert(format!("token_{}", i).into_bytes(), i as u32);
            }
            CoreBPE::new(vocab, merges, "bpe").unwrap()
        }),
    )
}

pub fn test_chat_template() -> Arc<ChatTemplate> {
    Arc::new(
        ChatTemplate::from_template_source("{{ system }}\n{{ user }}\n{{ assistant }}".to_string())
            .unwrap(),
    )
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
    ));
    (manager, buffer)
}

pub fn create_test_router() -> (Router, Arc<SlotManager<f16>>, Vec<usize>) {
    let (manager, buffer) = create_test_manager(4, 1000);
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
                slot.sequence_index += slot.filling_length;
                slot.filling_length = 0;
                slot.phase = Phase::Decode;
                slot.notify.notify_one();
            }
        });

        for &token_id in &generated_tokens {
            manager.batch_states.with_mut(|slots| {
                let slot = &mut slots[slot_index];
                let pos = slot.sequence_index;
                slot.sequence_index += 1;
                slot.token_count += 1;
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

pub async fn collect_body(body: Body) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut stream = body.into_data_stream();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }
    bytes
}
