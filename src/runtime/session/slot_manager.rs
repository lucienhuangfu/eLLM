use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::runtime::error::SlotError;
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::core::SlotState;
use crate::runtime::state::machine::SlotStateMachine;
use crate::runtime::state::types::Phase;

use super::types::{SessionHandle, SessionMode};

const LRU_SENTINEL: usize = usize::MAX;

pub struct SlotManager<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    slots: Arc<Mutex<Vec<SlotState>>>,
    active_prefill: Arc<Mutex<Vec<usize>>>,
    active_decode: Arc<Mutex<Vec<usize>>>,
    available_slots: Arc<Mutex<Vec<usize>>>,
    session_map: Arc<Mutex<HashMap<String, usize>>>,
    batch_sequences: Arc<crate::operators::send_sync_ptr::SharedMut<BatchSequence<T>>>,
    mode: SessionMode,
}

unsafe impl<T> Send for SlotManager<T> where T: Copy + crate::num_traits::FromNumber + Send {}

unsafe impl<T> Sync for SlotManager<T> where T: Copy + crate::num_traits::FromNumber + Sync {}

impl<T> SlotManager<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    pub fn new(
        num_slots: usize,
        batch_sequences: Arc<crate::operators::send_sync_ptr::SharedMut<BatchSequence<T>>>,
        mode: SessionMode,
    ) -> Self {
        let mut slots = Vec::with_capacity(num_slots);
        let mut available_slots = Vec::with_capacity(num_slots);

        for i in 0..num_slots {
            slots.push(SlotState::new_start_state());
            available_slots.push(i);
        }

        let mut slot_manager = Self {
            slots: Arc::new(Mutex::new(slots)),
            active_prefill: Arc::new(Mutex::new(Vec::new())),
            active_decode: Arc::new(Mutex::new(Vec::new())),
            available_slots: Arc::new(Mutex::new(available_slots)),
            session_map: Arc::new(Mutex::new(HashMap::new())),
            batch_sequences,
            mode,
        };

        slot_manager.init_lru();
        slot_manager
    }

    fn init_lru(&mut self) {
        let mut slots = self.slots.blocking_lock();
        let num_slots = slots.len();

        for i in 0..num_slots {
            slots[i].lru_prev = if i == 0 { LRU_SENTINEL } else { i - 1 };
            slots[i].lru_next = if i == num_slots - 1 {
                LRU_SENTINEL
            } else {
                i + 1
            };
        }
    }

    fn touch_lru(&self, slot_index: usize) {
        let mut slots = self.slots.blocking_lock();

        let prev = slots[slot_index].lru_prev;
        let next = slots[slot_index].lru_next;

        if prev != LRU_SENTINEL {
            slots[prev].lru_next = next;
        }
        if next != LRU_SENTINEL {
            slots[next].lru_prev = prev;
        }

        let head_prev = slots[0].lru_prev;

        slots[slot_index].lru_prev = LRU_SENTINEL;
        slots[slot_index].lru_next = head_prev;

        if head_prev != LRU_SENTINEL {
            slots[head_prev].lru_next = slot_index;
        }
        slots[0].lru_prev = slot_index;
    }

    fn evict_oldest(&self) -> usize {
        let mut slots = self.slots.blocking_lock();

        let mut tail = 0;
        while slots[tail].lru_next != LRU_SENTINEL {
            tail = slots[tail].lru_next;
        }

        let prev = slots[tail].lru_prev;
        if prev != LRU_SENTINEL {
            slots[prev].lru_next = LRU_SENTINEL;
        }

        tail
    }

    pub async fn acquire_session(&self, session_id: &str) -> Result<SessionHandle, SlotError> {
        let mut session_map = self.session_map.lock().await;

        if let Some(&slot_index) = session_map.get(session_id) {
            let mut slots = self.slots.lock().await;
            if let Some(entry) = slots.get_mut(slot_index) {
                entry.touch();
                self.touch_lru(slot_index);
                return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
            }
        }

        let slot_index = {
            let mut available = self.available_slots.lock().await;
            if !available.is_empty() {
                available.pop().unwrap()
            } else {
                self.evict_oldest()
            }
        };

        {
            let mut session_map = self.session_map.lock().await;
            let old_session_id: Option<String> = session_map
                .iter()
                .find(|(_, &idx)| idx == slot_index)
                .map(|(k, _)| k.clone());
            if let Some(id) = old_session_id {
                session_map.remove(&id);
            }
            session_map.insert(session_id.to_string(), slot_index);
        }

        {
            let mut slots = self.slots.lock().await;
            let entry = &mut slots[slot_index];
            entry.session_id = Some(session_id.to_string());
            entry.created_at = Instant::now();
            entry.last_accessed = Instant::now();
            entry.token_count = 0;
        }

        self.touch_lru(slot_index);

        Ok(SessionHandle::new(session_id.to_string(), slot_index))
    }

    pub async fn release_session(&self, session_id: &str, token_count: usize) {
        let mut session_map = self.session_map.lock().await;
        if let Some(&slot_index) = session_map.get(session_id) {
            let mut slots = self.slots.lock().await;
            if let Some(entry) = slots.get_mut(slot_index) {
                entry.token_count = token_count;

                if self.mode == SessionMode::NonReusable {
                    SlotStateMachine::reset_to_start(entry);
                    session_map.remove(session_id);

                    let mut available = self.available_slots.lock().await;
                    available.push(slot_index);
                }
            }
        }
    }

    pub async fn transition_to_prefill(
        &self,
        slot_index: usize,
        sequence_index: usize,
        filling_length: usize,
    ) {
        let mut slots = self.slots.lock().await;
        let entry = &mut slots[slot_index];

        let _ = SlotStateMachine::transition_to_prefill(entry, sequence_index, filling_length);

        self.remove_from_available(slot_index).await;
        self.add_to_active_prefill(slot_index).await;
    }

    pub async fn transition_to_decode(&self, slot_index: usize) {
        let mut slots = self.slots.lock().await;
        let entry = &mut slots[slot_index];

        let _ = SlotStateMachine::transition_to_decode(entry);

        self.remove_from_active_prefill(slot_index).await;
        self.add_to_active_decode(slot_index).await;
    }

    pub async fn transition_to_eos(&self, slot_index: usize) {
        let mut slots = self.slots.lock().await;
        let entry = &mut slots[slot_index];

        let _ = SlotStateMachine::transition_to_eos(entry);

        self.remove_from_active_prefill(slot_index).await;
        self.remove_from_active_decode(slot_index).await;
        self.add_to_available(slot_index).await;
    }

    pub async fn transition_to_timeout(&self, slot_index: usize) {
        let mut slots = self.slots.lock().await;
        let entry = &mut slots[slot_index];

        let _ = SlotStateMachine::transition_to_timeout(entry);

        self.remove_from_active_prefill(slot_index).await;
        self.remove_from_active_decode(slot_index).await;
    }

    pub async fn reset_to_start(&self, slot_index: usize) {
        let mut slots = self.slots.lock().await;
        let entry = &mut slots[slot_index];

        let old_phase = entry.phase;
        SlotStateMachine::reset_to_start(entry);

        if matches!(old_phase, Phase::Prefill) {
            self.remove_from_active_prefill(slot_index).await;
        } else if matches!(old_phase, Phase::Decode) {
            self.remove_from_active_decode(slot_index).await;
        }
        self.add_to_available(slot_index).await;
    }

    async fn add_to_active_prefill(&self, slot_index: usize) {
        let mut active = self.active_prefill.lock().await;
        if !active.contains(&slot_index) {
            active.push(slot_index);
        }
    }

    async fn remove_from_active_prefill(&self, slot_index: usize) {
        let mut active = self.active_prefill.lock().await;
        if let Some(pos) = active.iter().position(|&idx| idx == slot_index) {
            active.swap_remove(pos);
        }
    }

    async fn add_to_active_decode(&self, slot_index: usize) {
        let mut active = self.active_decode.lock().await;
        if !active.contains(&slot_index) {
            active.push(slot_index);
        }
    }

    async fn remove_from_active_decode(&self, slot_index: usize) {
        let mut active = self.active_decode.lock().await;
        if let Some(pos) = active.iter().position(|&idx| idx == slot_index) {
            active.swap_remove(pos);
        }
    }

    async fn add_to_available(&self, slot_index: usize) {
        let mut available = self.available_slots.lock().await;
        if !available.contains(&slot_index) {
            available.push(slot_index);
        }
    }

    async fn remove_from_available(&self, slot_index: usize) {
        let mut available = self.available_slots.lock().await;
        if let Some(pos) = available.iter().position(|&idx| idx == slot_index) {
            available.swap_remove(pos);
        }
    }

    pub fn get_active_prefill(&self) -> Vec<usize> {
        self.active_prefill.blocking_lock().clone()
    }

    pub fn get_active_decode(&self) -> Vec<usize> {
        self.active_decode.blocking_lock().clone()
    }

    pub async fn has_work(&self) -> bool {
        !self.active_prefill.lock().await.is_empty() || !self.active_decode.lock().await.is_empty()
    }

    pub async fn get_slot(&self, slot_index: usize) -> Option<SlotState> {
        let slots = self.slots.lock().await;
        slots.get(slot_index).cloned()
    }

    pub async fn advance_sequence(&self, slot_index: usize, steps: usize) {
        let mut slots = self.slots.lock().await;
        if let Some(entry) = slots.get_mut(slot_index) {
            let phase_change = SlotStateMachine::advance_sequence(entry, steps);

            if phase_change == Some(Phase::Decode) {
                self.remove_from_active_prefill(slot_index).await;
                self.add_to_active_decode(slot_index).await;
            }
        }
    }

    pub async fn get_cached_tokens(&self, session_id: &str) -> Option<(usize, usize)> {
        let session_map = self.session_map.lock().await;
        let &slot_index = session_map.get(session_id)?;
        let slots = self.slots.lock().await;
        let entry = slots.get(slot_index)?;
        if entry.token_count > 0 {
            Some((slot_index, entry.token_count))
        } else {
            None
        }
    }

    pub async fn calculate_delta(
        &self,
        session_id: &str,
        new_tokens: &[u32],
    ) -> Option<(usize, Vec<u32>)> {
        let (slot_index, cached_count) = self.get_cached_tokens(session_id).await?;

        let cached_tokens = self
            .batch_sequences
            .with(|batch_seq| batch_seq.token_ids(slot_index, 0, cached_count));

        let min_len = cached_tokens.len().min(new_tokens.len());
        let mut prefix_len = 0;

        while prefix_len < min_len && cached_tokens[prefix_len] == new_tokens[prefix_len] {
            prefix_len += 1;
        }

        if prefix_len > 0 {
            Some((prefix_len, new_tokens[prefix_len..].to_vec()))
        } else {
            None
        }
    }

    pub async fn session_count(&self) -> usize {
        self.session_map.lock().await.len()
    }

    pub fn total_slots(&self) -> usize {
        self.slots.blocking_lock().len()
    }
}