use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;

use super::types::Phase;

const LRU_SENTINEL: usize = usize::MAX;

#[derive(Clone)]
pub struct SlotState {
    pub sequence_index: usize,
    pub kv_index: usize,
    pub filling_length: usize,
    pub phase: Phase,

    pub session_id: Option<String>,
    pub token_count: usize,

    pub created_at: Instant,
    pub last_accessed: Instant,

    pub(crate) notify: Arc<Notify>,

    pub(crate) lru_prev: usize,
    pub(crate) lru_next: usize,
}

impl SlotState {
    pub fn new_start_state() -> Self {
        Self {
            sequence_index: usize::MAX,
            kv_index: usize::MAX,
            filling_length: 0,
            phase: Phase::Start,
            session_id: None,
            token_count: 0,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            notify: Arc::new(Notify::new()),
            lru_prev: LRU_SENTINEL,
            lru_next: LRU_SENTINEL,
        }
    }

    pub fn new_prefill_state(sequence_index: usize, filling_length: usize) -> Self {
        Self {
            sequence_index,
            kv_index: sequence_index,
            filling_length,
            phase: Phase::Prefill,
            session_id: None,
            token_count: 0,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            notify: Arc::new(Notify::new()),
            lru_prev: LRU_SENTINEL,
            lru_next: LRU_SENTINEL,
        }
    }

    pub fn new_decode_state(sequence_index: usize, kv_index: usize) -> Self {
        Self {
            sequence_index,
            kv_index,
            filling_length: 0,
            phase: Phase::Decode,
            session_id: None,
            token_count: 0,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            notify: Arc::new(Notify::new()),
            lru_prev: LRU_SENTINEL,
            lru_next: LRU_SENTINEL,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self.phase, Phase::Prefill | Phase::Decode)
    }

    pub fn is_available(&self) -> bool {
        matches!(self.phase, Phase::Start | Phase::Eos)
    }

    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }

    pub fn notify(&self) -> Arc<Notify> {
        Arc::clone(&self.notify)
    }
}

impl Default for SlotState {
    fn default() -> Self {
        Self::new_start_state()
    }
}