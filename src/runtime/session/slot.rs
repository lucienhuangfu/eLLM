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
    pub next_sequence_index: usize,
    pub prompt_length: usize,
    pub phase: Phase,
    pub sequence_length: usize,
    pub(crate) notify: Arc<Notify>,
}

impl SlotState {
    pub fn idle() -> Self {
        Self {
            next_sequence_index: usize::MAX,
            prompt_length: usize::MAX,
            phase: Phase::Start,
            sequence_length: 0,
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn start_prefill(&mut self, start_index: usize, filling_length: usize) {
        self.next_sequence_index = start_index;
        self.prompt_length = start_index + filling_length;
        self.phase = Phase::Prefill;
        self.sequence_length = filling_length;
    }

    pub fn start_decode(&mut self, next_sequence_index: usize, prompt_length: usize) {
        self.next_sequence_index = next_sequence_index;
        self.prompt_length = prompt_length;
        self.phase = Phase::Decode;
    }

    pub fn is_available(&self) -> bool {
        matches!(self.phase, Phase::Start | Phase::Eos)
    }

    pub fn filling_length(&self) -> usize {
        self.prompt_length.saturating_sub(self.next_sequence_index)
    }

    pub fn reset_to_start(&mut self) {
        self.next_sequence_index = usize::MAX;
        self.prompt_length = usize::MAX;
        self.phase = Phase::Start;
        self.sequence_length = 0;
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
}

impl SessionHandle {
    pub fn new(session_id: String, slot_index: usize) -> Self {
        Self {
            session_id,
            slot_index,
        }
    }
}
