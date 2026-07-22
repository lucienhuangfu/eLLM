use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use tokio::sync::Mutex as TokioMutex;

use crate::num_traits::FromNumber;
use crate::operators::send_sync_ptr::SharedMut;
use super::batch_sequence::BatchSequence;

use super::lru::LruList;
use super::types::{SessionHandle, SessionMode, SlotResult, SlotState};

// ── ReservedSlot ───────────────────────────────────────────

struct ReservedSlot {
    slot_index: usize,
    cancel_flag: Arc<AtomicBool>,
}

// ── SlotManager ────────────────────────────────────────────

pub struct SlotManager<T: Copy + FromNumber> {
    batch_states: Arc<SharedMut<Vec<SlotState>>>,
    batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    lru: Arc<StdMutex<LruList>>,
    session_map: TokioMutex<HashMap<String, usize>>,
    reserved_slots: Arc<TokioMutex<HashMap<String, ReservedSlot>>>,
    mode: SessionMode,
    reuse_timeout: Duration,
}

impl<T: Copy + FromNumber> SlotManager<T> {
    pub fn new(
        num_slots: usize,
        batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
        batch_states: Arc<SharedMut<Vec<SlotState>>>,
        mode: SessionMode,
        reuse_timeout_ms: u64,
    ) -> Self {
        Self {
            batch_states,
            batch_sequences,
            lru: Arc::new(StdMutex::new(LruList::new(num_slots))),
            session_map: TokioMutex::new(HashMap::new()),
            reserved_slots: Arc::new(TokioMutex::new(HashMap::new())),
            mode,
            reuse_timeout: Duration::from_millis(reuse_timeout_ms),
        }
    }

    pub fn batch_list(&self) -> Arc<SharedMut<Vec<SlotState>>> {
        Arc::clone(&self.batch_states)
    }

    pub(crate) fn batch_sequences(&self) -> Arc<SharedMut<BatchSequence<T>>> {
        Arc::clone(&self.batch_sequences)
    }

    pub fn with_slots<R>(&self, f: impl FnOnce(&[SlotState]) -> R) -> R {
        self.batch_states.with(|v| f(v.as_slice()))
    }

    pub fn with_slots_mut<R>(&self, f: impl FnOnce(&mut [SlotState]) -> R) -> R {
        self.batch_states.with_mut(|v| f(v.as_mut_slice()))
    }

    pub fn with_sequence<R>(&self, f: impl FnOnce(&BatchSequence<T>) -> R) -> R {
        self.batch_sequences.with(f)
    }

    pub fn with_sequence_mut<R>(&self, f: impl FnOnce(&mut BatchSequence<T>) -> R) -> R {
        self.batch_sequences.with_mut(f)
    }

    pub fn with_slot_and_sequence_mut<R>(
        &self,
        slot_index: usize,
        f: impl FnOnce(&mut SlotState, &mut BatchSequence<T>) -> R,
    ) -> R {
        self.batch_states.with_mut(|slots| {
            self.batch_sequences.with_mut(|seq| {
                f(&mut slots[slot_index], seq)
            })
        })
    }

    pub fn decode_single_token(&self, slot_index: usize, token_index: usize) -> String {
        self.batch_sequences.with(|seq| {
            seq.decode_single_token(slot_index, token_index)
                .unwrap_or_default()
        })
    }

    pub fn decode_generated_text(&self, slot_index: usize) -> String {
        self.batch_states.with(|slots| {
            let record = &slots[slot_index];
            self.batch_sequences
                .with(|seq| seq.decode_generated_text(slot_index, record))
        })
    }

    pub fn detach_from_lru(&self, idx: usize) {
        self.lru.lock().unwrap().remove(idx);
    }

    pub async fn acquire_session(&self, session_id: &str) -> SlotResult<SessionHandle> {
        let mut map = self.session_map.lock().await;

        if let Some(&slot_index) = map.get(session_id) {
            return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
        }

        let reserved = self.reserved_slots.lock().await.remove(session_id);

        if let Some(r) = reserved {
            r.cancel_flag.store(true, Ordering::Release);
            map.insert(session_id.to_string(), r.slot_index);
            return Ok(SessionHandle::reused(session_id.to_string(), r.slot_index));
        }

        let slot_index = self.lru.lock().unwrap().pop_back();

        if let Some(old_id) = map
            .iter()
            .find(|(_, &idx)| idx == slot_index)
            .map(|(k, _)| k.clone())
        {
            map.remove(&old_id);
        }
        map.insert(session_id.to_string(), slot_index);

        self.batch_states.with_mut(|slots| {
            slots[slot_index] = SlotState::new_start_state();
        });

        Ok(SessionHandle::new(session_id.to_string(), slot_index))
    }

    pub async fn release_session(&self, session_id: &str, token_count: usize) {
        let mut map = self.session_map.lock().await;
        let Some(&slot_index) = map.get(session_id) else {
            return;
        };

        self.batch_states.with_mut(|slots| {
            slots[slot_index].token_count = token_count;
        });

        if self.mode == SessionMode::NonReusable {
            self.batch_states.with_mut(|slots| {
                slots[slot_index].reset_to_start();
            });
            map.remove(session_id);
            self.lru.lock().unwrap().push_front(slot_index);
            return;
        }

        let session_id_owned = session_id.to_string();
        map.remove(session_id);

        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.reserved_slots.lock().await.insert(
            session_id_owned.clone(),
            ReservedSlot {
                slot_index,
                cancel_flag: Arc::clone(&cancel_flag),
            },
        );

        let reserved_slots = self.reserved_slots.clone();
        let batch_states = Arc::clone(&self.batch_states);
        let lru = Arc::clone(&self.lru);
        let timeout = self.reuse_timeout;

        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            if cancel_flag.load(Ordering::Acquire) {
                return;
            }

            let mut reserved = reserved_slots.lock().await;
            if let Some(r) = reserved.remove(&session_id_owned) {
                batch_states.with_mut(|slots| {
                    slots[r.slot_index].reset_to_start();
                });
                lru.lock().unwrap().push_front(r.slot_index);
            }
        });
    }

    pub async fn get_prefix_match_len(
        &self,
        session_id: &str,
        new_tokens: &[u32],
    ) -> Option<usize> {
        let map = self.session_map.lock().await;
        let &slot_index = map.get(session_id)?;
        let cached_count = self.with_slots(|slots| slots[slot_index].token_count);
        if cached_count == 0 {
            return None;
        }
        drop(map);

        let cached_tokens = self
            .batch_sequences
            .with(|batch_seq| batch_seq.token_ids(slot_index, 0, cached_count));

        let min_len = cached_tokens.len().min(new_tokens.len());
        let prefix_len = (0..min_len)
            .take_while(|&i| cached_tokens[i] == new_tokens[i])
            .count();

        if prefix_len > 0 {
            Some(prefix_len)
        } else {
            None
        }
    }
}

// ── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::session::batch_sequence::BatchSequence;
    use crate::runtime::session::types::Phase;
    use crate::num_traits::FromNumber;

    fn create_test_manager_with_buffer(
        batch_size: usize,
        timeout_ms: u64,
        buffer: &mut Vec<usize>,
    ) -> Arc<SlotManager<f16>> {
        let seq_len = 1024;
        buffer.resize(batch_size * seq_len, 0);
        let batch_sequences = Arc::new(SharedMut::new(BatchSequence::<f16> {
            sequences: buffer.as_mut_ptr(),
            batch_temperature: vec![<f16 as FromNumber>::from_f32(1.0); batch_size],
            row_size: batch_size,
            col_size: seq_len,
            tokenizer: BatchSequence::<f16>::default().tokenizer,
            chat_template: BatchSequence::<f16>::default().chat_template,
        }));
        let batch_states = Arc::new(SharedMut::new(
            (0..batch_size)
                .map(|_| SlotState::new_start_state())
                .collect::<Vec<_>>(),
        ));
        Arc::new(SlotManager::new(
            batch_size,
            batch_sequences,
            batch_states,
            SessionMode::Reusable,
            timeout_ms,
        ))
    }

    #[tokio::test]
    async fn test_get_prefix_match_len_session_not_found() {
        let mut buffer = Vec::new();
        let manager = create_test_manager_with_buffer(4, 1000, &mut buffer);
        let delta = manager.get_prefix_match_len("nonexistent", &[1, 2, 3]).await;
        assert!(delta.is_none());
    }
}
