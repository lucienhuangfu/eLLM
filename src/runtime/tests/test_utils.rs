use std::sync::Arc;

use crate::num_traits::FromNumber;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::session::BatchSequence;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{Phase, SessionMode, SlotManager, SlotState};

pub fn advance_slot(slot: &mut SlotState, steps: usize) -> Option<Phase> {
    if slot.phase == Phase::Eos {
        return None;
    }
    slot.sequence_index += steps;
    if slot.phase == Phase::Prefill {
        slot.filling_length = slot.filling_length.saturating_sub(steps);
        if slot.filling_length == 0 {
            slot.phase = Phase::Decode;
            return Some(Phase::Decode);
        }
    }
    None
}

pub fn make_batch_list(slots: Vec<SlotState>) -> Arc<SharedMut<Vec<SlotState>>> {
    Arc::new(SharedMut::new(slots))
}

pub fn create_test_manager(
    batch_size: usize,
    timeout_ms: u64,
) -> (Arc<SlotManager<f16>>, Vec<usize>) {
    let seq_len = 1024;
    let mut buffer = vec![0usize; batch_size * seq_len];
    let batch_sequences = Arc::new(SharedMut::new(
        BatchSequence::<f16> {
            sequences: buffer.as_mut_ptr(),
            batch_temperature: vec![<f16 as FromNumber>::from_f32(1.0); batch_size],
            row_size: batch_size,
            col_size: seq_len,
            tokenizer: BatchSequence::<f16>::default().tokenizer,
            chat_template: BatchSequence::<f16>::default().chat_template,
        },
    ));
    let batch_states = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
            .collect::<Vec<_>>(),
    ));
    let manager = Arc::new(SlotManager::new(
        batch_size,
        batch_sequences,
        batch_states,
        SessionMode::Reusable,
        timeout_ms,
    ));
    (manager, buffer)
}

pub fn create_test_manager_with_mode(
    batch_size: usize,
    timeout_ms: u64,
    mode: SessionMode,
) -> (Arc<SlotManager<f16>>, Vec<usize>) {
    let seq_len = 1024;
    let mut buffer = vec![0usize; batch_size * seq_len];
    let batch_sequences = Arc::new(SharedMut::new(
        BatchSequence::<f16> {
            sequences: buffer.as_mut_ptr(),
            batch_temperature: vec![<f16 as FromNumber>::from_f32(1.0); batch_size],
            row_size: batch_size,
            col_size: seq_len,
            tokenizer: BatchSequence::<f16>::default().tokenizer,
            chat_template: BatchSequence::<f16>::default().chat_template,
        },
    ));
    let batch_states = Arc::new(SharedMut::new(
        (0..batch_size)
            .map(|_| SlotState::new_start_state())
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

pub fn run_prefill_and_decode(
    manager: &SlotManager<f16>,
    scheduler: &Scheduler,
    slot_index: usize,
    prefill_len: usize,
    decode_steps: usize,
) {
    manager.with_slots_mut(|slots| {
        slots[slot_index] = SlotState::new_prefill_state(0, prefill_len);
    });

    assert!(scheduler.schedule_batch());

    manager.with_slots_mut(|slots| {
        advance_slot(&mut slots[slot_index], prefill_len);
    });

    for _ in 0..decode_steps {
        assert!(scheduler.schedule_batch());
        manager.with_slots_mut(|slots| {
            advance_slot(&mut slots[slot_index], 1);
        });
    }

    manager.with_slots_mut(|slots| {
        slots[slot_index].phase = Phase::Eos;
    });
}
