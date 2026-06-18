use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;

use crate::runtime::state::types::Phase;

#[derive(Clone)]
pub struct SlotEntry {
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

impl SlotEntry {
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
            lru_prev: usize::MAX,
            lru_next: usize::MAX,
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
            lru_prev: usize::MAX,
            lru_next: usize::MAX,
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
            lru_prev: usize::MAX,
            lru_next: usize::MAX,
        }
    }

    pub fn transition_to_decode(&mut self) {
        self.phase = Phase::Decode;
        self.filling_length = 0;
        self.notify.notify_one();
    }

    pub fn transition_to_eos(&mut self) {
        self.phase = Phase::Eos;
        self.notify.notify_one();
    }

    pub fn transition_to_timeout(&mut self) {
        self.phase = Phase::Timeout;
        self.notify.notify_one();
    }

    pub fn reset_to_start(&mut self) {
        self.sequence_index = usize::MAX;
        self.kv_index = usize::MAX;
        self.filling_length = 0;
        self.phase = Phase::Start;
        self.session_id = None;
        self.token_count = 0;
    }

    pub fn is_active(&self) -> bool {
        matches!(self.phase, Phase::Prefill | Phase::Decode)
    }

    pub fn is_available(&self) -> bool {
        matches!(self.phase, Phase::Start | Phase::Eos)
    }

    pub fn advance_sequence(&mut self, steps: usize) {
        self.sequence_index += steps;
        if self.phase == Phase::Prefill {
            self.filling_length = self.filling_length.saturating_sub(steps);
            if self.filling_length == 0 {
                self.transition_to_decode();
            }
        }
    }

    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }

    pub fn notify(&self) -> Arc<Notify> {
        Arc::clone(&self.notify)
    }
}
