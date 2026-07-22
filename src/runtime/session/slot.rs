use std::sync::Arc;

use tokio::sync::Notify;

// ── Phase ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Phase {
    Start,
    Prefill,
    Decode,
    Eos,
}

// ── SlotState ──────────────────────────────────────────────

pub struct SlotState {
    pub sequence_index: usize,
    pub kv_index: usize,
    pub filling_length: usize,
    pub phase: Phase,
    pub token_count: usize,
    pub(crate) notify: Arc<Notify>,
}

impl SlotState {
    pub fn idle() -> Self {
        Self {
            sequence_index: usize::MAX,
            kv_index: usize::MAX,
            filling_length: 0,
            phase: Phase::Start,
            token_count: 0,
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn start_prefill(&mut self, sequence_index: usize, filling_length: usize) {
        self.sequence_index = sequence_index;
        self.kv_index = sequence_index;
        self.filling_length = filling_length;
        self.phase = Phase::Prefill;
        self.token_count = filling_length;
    }

    pub fn start_decode(&mut self, sequence_index: usize, kv_index: usize) {
        self.sequence_index = sequence_index;
        self.kv_index = kv_index;
        self.filling_length = 0;
        self.phase = Phase::Decode;
    }

    pub fn is_available(&self) -> bool {
        matches!(self.phase, Phase::Start | Phase::Eos)
    }

    pub fn reset_to_start(&mut self) {
        self.sequence_index = usize::MAX;
        self.kv_index = usize::MAX;
        self.filling_length = 0;
        self.phase = Phase::Start;
        self.token_count = 0;
    }
}

// ── SessionMode ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionMode {
    Reusable,
    NonReusable,
}

// ── SessionHandle ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SessionHandle {
    pub session_id: String,
    pub slot_index: usize,
    pub is_reused: bool,
}

impl SessionHandle {
    pub fn new(session_id: String, slot_index: usize, is_reused: bool) -> Self {
        Self {
            session_id,
            slot_index,
            is_reused,
        }
    }
}
