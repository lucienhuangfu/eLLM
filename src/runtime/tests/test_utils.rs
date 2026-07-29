use std::sync::Arc;

use crate::num_traits::FromNumber;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::loader::ChatTemplate;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::BatchSequence;
use crate::runtime::session::{Phase, SessionMode, SlotManager, SlotState};
use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;

pub fn advance_slot(slot: &mut SlotState, steps: usize) -> Option<Phase> {
    if slot.phase == Phase::Eos {
        return None;
    }
    slot.next_sequence_index += steps;
    if slot.phase == Phase::Prefill {
        if slot.filling_length() == 0 {
            slot.phase = Phase::Decode;
            return Some(Phase::Decode);
        }
    } else {
        slot.sequence_length += steps;
    }
    None
}

pub fn make_slot_list(slots: Vec<SlotState>) -> Arc<SharedMut<Vec<SlotState>>> {
    Arc::new(SharedMut::new(slots))
}

pub fn make_prefill_state(next_sequence_index: usize, filling_length: usize) -> SlotState {
    let mut s = SlotState::idle();
    s.start_prefill(next_sequence_index, filling_length);
    s
}

pub fn make_decode_state(next_sequence_index: usize, prompt_length: usize) -> SlotState {
    let mut s = SlotState::idle();
    s.start_decode(next_sequence_index, prompt_length);
    s
}

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
        true,
        true,
    ));
    (manager, buffer)
}

pub fn run_prefill_and_decode(
    manager: &SlotManager<f16>,
    scheduler: &Scheduler,
    slot_index: usize,
    prefill_length: usize,
    decode_steps: usize,
) {
    manager.batch_states.with_mut(|slots| {
        slots[slot_index].start_prefill(0, prefill_length);
    });

    assert!(scheduler.schedule_batch());

    manager.batch_states.with_mut(|slots| {
        advance_slot(&mut slots[slot_index], prefill_length);
    });

    for _ in 0..decode_steps {
        assert!(scheduler.schedule_batch());
        manager.batch_states.with_mut(|slots| {
            advance_slot(&mut slots[slot_index], 1);
        });
    }

    manager.batch_states.with_mut(|slots| {
        slots[slot_index].phase = Phase::Eos;
    });
}
