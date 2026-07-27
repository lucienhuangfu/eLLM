use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use tokio::sync::{Mutex as TokioMutex, Notify};

use super::sequence::BatchSequence;
use super::slot::{Phase, SessionHandle, SessionMode, SlotState};
use crate::num_traits::FromNumber;
use crate::operators::send_sync_ptr::SharedMut;
use crate::serving::{ApiError, ApiResult, ChatMessage};

// ── SlotManager ────────────────────────────────────────────

pub struct SlotManager<T: Copy + FromNumber> {
    pub(crate) batch_states: Arc<SharedMut<Vec<SlotState>>>,
    pub(crate) batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    lru: StdMutex<Vec<usize>>,
    session_map: TokioMutex<HashMap<String, usize>>,
    reserved_slots: TokioMutex<HashMap<String, (usize, Arc<AtomicBool>)>>,
    mode: SessionMode,
    reuse_timeout: Duration,
}

impl<T: Copy + FromNumber + Send + Sync + 'static> SlotManager<T> {
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
            lru: StdMutex::new((0..num_slots).collect()),
            session_map: TokioMutex::new(HashMap::new()),
            reserved_slots: TokioMutex::new(HashMap::new()),
            mode,
            reuse_timeout: Duration::from_millis(reuse_timeout_ms),
        }
    }

    // ── Session lifecycle ─────────────────────────────────

    pub async fn acquire_session(&self, session_id: &str) -> ApiResult<SessionHandle> {
        let mut map = self.session_map.lock().await;

        if let Some(&slot_index) = map.get(session_id) {
            return Ok(SessionHandle::new(session_id.to_string(), slot_index));
        }

        // 尝试回收 reserved slot
        if let Some((slot_index, cancel_flag)) = self.reserved_slots.lock().await.remove(session_id)
        {
            cancel_flag.store(true, Ordering::Release);
            map.insert(session_id.to_string(), slot_index);
            return Ok(SessionHandle::new(session_id.to_string(), slot_index));
        }

        // 从 LRU 尾部淘汰
        let mut lru = self.lru.lock().unwrap();
        let slot_index = lru
            .pop()
            .ok_or_else(|| ApiError::SlotUnavailable("all slots are occupied, please retry later".into()))?;
        drop(lru);

        if let Some(old_id) = map
            .iter()
            .find(|(_, &idx)| idx == slot_index)
            .map(|(k, _)| k.clone())
        {
            map.remove(&old_id);
        }
        map.insert(session_id.to_string(), slot_index);

        self.batch_states.with_mut(|slots| {
            slots[slot_index] = SlotState::idle();
        });

        Ok(SessionHandle::new(session_id.to_string(), slot_index))
    }

    pub async fn release_session(self: Arc<Self>, session_id: &str, sequence_length: usize) {
        let mut map = self.session_map.lock().await;
        let Some(&slot_index) = map.get(session_id) else {
            return;
        };

        self.batch_states.with_mut(|slots| {
            slots[slot_index].sequence_length = sequence_length;
        });

        if self.mode == SessionMode::NonReusable {
            self.batch_states
                .with_mut(|slots| slots[slot_index].reset_to_start());
            map.remove(session_id);
            self.lru_remove_and_push_front(slot_index);
            return;
        }

        // Reusable 模式：放入 reserved，超时后回收
        let session_id_owned = session_id.to_string();
        map.remove(session_id);
        drop(map);

        let cancel_flag = Arc::new(AtomicBool::new(false));
        {
            let mut reserved = self.reserved_slots.lock().await;
            reserved.insert(
                session_id_owned.clone(),
                (slot_index, Arc::clone(&cancel_flag)),
            );
        }

        let timeout = self.reuse_timeout;

        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            if cancel_flag.load(Ordering::Acquire) {
                return;
            }
            let idx = {
                let mut reserved = self.reserved_slots.lock().await;
                reserved.remove(&session_id_owned).map(|(idx, _)| idx)
            };
            if let Some(idx) = idx {
                self.batch_states
                    .with_mut(|slots| slots[idx].reset_to_start());
                self.lru_remove_and_push_front(idx);
            }
        });
    }

    // ── Prompt writing ────────────────────────────────────

    pub async fn write_prompts(
        &self,
        slot_index: usize,
        session_id: &str,
        messages: &[ChatMessage],
        temperature: Option<f32>,
    ) -> ApiResult<(usize, Arc<Notify>)> {
        let message_pairs: Vec<(&str, &str)> = messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str()))
            .collect();

        let new_tokens: Vec<u32> = self
            .batch_sequences
            .with(|seq| seq.tokenize_messages(&message_pairs))
            .map_err(ApiError::TokenizationError)?;

        let prefix_len = if self.mode == SessionMode::Reusable {
            self.prefix_match_len(session_id, &new_tokens)
                .await
                .unwrap_or(0)
        } else {
            0
        };

        self.ensure_slot_available(slot_index)?;
        self.lru_remove(slot_index);

        let temperature = temperature.unwrap_or(1.0);
        let remaining_tokens = &new_tokens[prefix_len..];

        self.batch_states.with_mut(|slots| {
            self.batch_sequences.with_mut(|seq| {
                let record = &mut slots[slot_index];
                seq.write_tokens_at(slot_index, prefix_len, remaining_tokens, temperature)
                    .map(|write_len| {
                        let total_prompt_length = prefix_len + write_len;
                        record.next_sequence_index = prefix_len;
                        record.prompt_length = total_prompt_length;
                        record.phase = Phase::Prefill;
                        (write_len, record.notify.clone())
                    })
                    .map_err(ApiError::TokenizationError)
            })
        })
    }

    // ── Decode helpers ────────────────────────────────────

    pub fn decode_single_token(&self, slot_index: usize, token_index: usize) -> String {
        self.batch_sequences.with(|seq| {
            seq.decode_single_token(slot_index, token_index)
                .unwrap_or_default()
        })
    }

    pub fn decode_token_span(&self, slot_index: usize, begin: usize, end: usize) -> String {
        self.batch_sequences
            .with(|seq| seq.decode_token_span(slot_index, begin, end))
    }

    pub fn decode_generated_text(&self, slot_index: usize) -> String {
        self.batch_states.with(|slots| {
            let record = &slots[slot_index];
            self.batch_sequences.with(|seq| {
                seq.decode_token_span(slot_index, record.prompt_length, record.next_sequence_index)
            })
        })
    }

    pub fn is_eos(&self, slot_index: usize) -> bool {
        self.batch_states
            .with(|slots| matches!(slots[slot_index].phase, Phase::Eos))
    }

    pub fn get_token_index_and_phase(&self, slot_index: usize) -> (usize, Phase) {
        self.batch_states
            .with(|slots| (slots[slot_index].next_sequence_index, slots[slot_index].phase))
    }

    pub fn get_next_sequence_index(&self, slot_index: usize) -> usize {
        self.batch_states
            .with(|slots| slots[slot_index].next_sequence_index)
    }

    pub fn get_prompt_length(&self, slot_index: usize) -> usize {
        self.batch_states
            .with(|slots| slots[slot_index].prompt_length)
    }

    // ── Private helpers ───────────────────────────────────

    fn ensure_slot_available(&self, slot_index: usize) -> ApiResult<()> {
        self.batch_states.with(|slots| {
            if slots[slot_index].is_available() {
                Ok(())
            } else {
                Err(ApiError::SlotUnavailable(
                    "slot is not in Start or Eos phase".into(),
                ))
            }
        })
    }

    fn lru_remove(&self, idx: usize) {
        let mut lru = self.lru.lock().unwrap();
        if let Some(pos) = lru.iter().position(|&x| x == idx) {
            lru.remove(pos);
        }
    }

    fn lru_remove_and_push_front(&self, idx: usize) {
        let mut lru = self.lru.lock().unwrap();
        if let Some(pos) = lru.iter().position(|&x| x == idx) {
            lru.remove(pos);
        }
        lru.insert(0, idx);
    }

    pub(crate) async fn prefix_match_len(
        &self,
        session_id: &str,
        new_tokens: &[u32],
    ) -> Option<usize> {
        let map = self.session_map.lock().await;
        let &slot_index = map.get(session_id)?;
        let cached_count = self
            .batch_states
            .with(|slots| slots[slot_index].sequence_length);
        if cached_count == 0 {
            return None;
        }
        drop(map);

        let cached_tokens = self
            .batch_sequences
            .with(|seq| seq.token_ids(slot_index, 0, cached_count));

        let prefix_len = cached_tokens
            .iter()
            .zip(new_tokens)
            .take_while(|(a, b)| a == b)
            .count();

        (prefix_len > 0).then_some(prefix_len)
    }
}
