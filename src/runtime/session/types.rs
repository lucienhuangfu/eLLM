use std::fmt;
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
    fn fresh(phase: Phase, sequence_index: usize, kv_index: usize, filling_length: usize) -> Self {
        Self {
            sequence_index,
            kv_index,
            filling_length,
            phase,
            token_count: 0,
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn new_start_state() -> Self {
        Self::fresh(Phase::Start, usize::MAX, usize::MAX, 0)
    }

    pub fn new_prefill_state(sequence_index: usize, filling_length: usize) -> Self {
        Self::fresh(
            Phase::Prefill,
            sequence_index,
            sequence_index,
            filling_length,
        )
    }

    pub fn new_decode_state(sequence_index: usize, kv_index: usize) -> Self {
        Self::fresh(Phase::Decode, sequence_index, kv_index, 0)
    }

    pub fn is_available(&self) -> bool {
        matches!(self.phase, Phase::Start | Phase::Eos)
    }

    pub fn notify(&self) -> Arc<Notify> {
        Arc::clone(&self.notify)
    }

    pub fn reset_to_start(&mut self) {
        self.sequence_index = usize::MAX;
        self.kv_index = usize::MAX;
        self.filling_length = 0;
        self.phase = Phase::Start;
        self.token_count = 0;
    }
}

impl Default for SlotState {
    fn default() -> Self {
        Self::new_start_state()
    }
}

// ── SlotError ──────────────────────────────────────────────

#[derive(Debug)]
pub enum SlotError {
    AllocatorUnavailable,
    SlotQueueEmpty,
    SlotNotFound,
}

impl fmt::Display for SlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlotError::AllocatorUnavailable => write!(f, "Slot allocator unavailable"),
            SlotError::SlotQueueEmpty => write!(f, "Slot queue empty while permit acquired"),
            SlotError::SlotNotFound => write!(f, "Slot not found"),
        }
    }
}

impl std::error::Error for SlotError {}

pub type SlotResult<T> = Result<T, SlotError>;

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
    pub fn new(session_id: String, slot_index: usize) -> Self {
        Self {
            session_id,
            slot_index,
            is_reused: false,
        }
    }

    pub fn reused(session_id: String, slot_index: usize) -> Self {
        Self {
            session_id,
            slot_index,
            is_reused: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_repr_and_ordering() {
        assert_eq!(std::mem::size_of::<Phase>(), 1);
        assert!((Phase::Start as u8) < (Phase::Prefill as u8));
        assert!((Phase::Prefill as u8) < (Phase::Decode as u8));
        assert!((Phase::Decode as u8) < (Phase::Eos as u8));
    }

    #[test]
    fn test_slot_state_constructors() {
        let state = SlotState::new_start_state();
        assert_eq!(state.phase, Phase::Start);
        assert!(state.is_available());
        assert_eq!(state.sequence_index, usize::MAX);

        let state = SlotState::new_prefill_state(0, 10);
        assert_eq!(state.phase, Phase::Prefill);
        assert_eq!(state.sequence_index, 0);
        assert_eq!(state.kv_index, 0);
        assert_eq!(state.filling_length, 10);

        let state = SlotState::new_decode_state(5, 5);
        assert_eq!(state.phase, Phase::Decode);
        assert_eq!(state.sequence_index, 5);
        assert_eq!(state.kv_index, 5);
        assert_eq!(state.filling_length, 0);
    }

    #[test]
    fn test_reset_to_start() {
        let mut state = SlotState::new_decode_state(5, 5);
        state.sequence_index = 42;
        state.token_count = 100;
        state.reset_to_start();
        assert_eq!(state.phase, Phase::Start);
        assert_eq!(state.sequence_index, usize::MAX);
        assert_eq!(state.filling_length, 0);
        assert_eq!(state.token_count, 0);
    }

    #[test]
    fn test_is_available() {
        assert!(SlotState::new_start_state().is_available());
        let mut s = SlotState::new_prefill_state(0, 10);
        assert!(!s.is_available());
        s.phase = Phase::Decode;
        assert!(!s.is_available());
        s.phase = Phase::Eos;
        assert!(s.is_available());
    }

    #[test]
    fn test_slot_error_display() {
        assert_eq!(
            SlotError::AllocatorUnavailable.to_string(),
            "Slot allocator unavailable"
        );
        assert_eq!(
            SlotError::SlotQueueEmpty.to_string(),
            "Slot queue empty while permit acquired"
        );
        assert_eq!(SlotError::SlotNotFound.to_string(), "Slot not found");
    }

    #[test]
    fn test_slot_error_propagation() {
        fn inner() -> SlotResult<()> {
            Err(SlotError::SlotNotFound)
        }
        fn outer() -> SlotResult<i32> {
            inner().map(|_| 42)
        }
        assert!(matches!(outer(), Err(SlotError::SlotNotFound)));
    }

    #[test]
    fn test_session_handle_constructors() {
        let h1 = SessionHandle::new("test".to_string(), 5);
        assert_eq!(h1.session_id, "test");
        assert_eq!(h1.slot_index, 5);
        assert!(!h1.is_reused);

        let h2 = SessionHandle::reused("test".to_string(), 5);
        assert_eq!(h2.session_id, "test");
        assert_eq!(h2.slot_index, 5);
        assert!(h2.is_reused);
    }
}
