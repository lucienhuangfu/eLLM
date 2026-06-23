use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;

use crate::runtime::state::types::Phase;

const LRU_SENTINEL: usize = usize::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionError {
    InvalidTransition,
    AlreadyInTargetState,
}

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

    pub fn transition_to_prefill(
        &mut self,
        sequence_index: usize,
        filling_length: usize,
    ) -> Result<(), TransitionError> {
        if self.phase == Phase::Prefill {
            return Err(TransitionError::AlreadyInTargetState);
        }

        if !matches!(self.phase, Phase::Start | Phase::Eos | Phase::Timeout) {
            return Err(TransitionError::InvalidTransition);
        }

        self.sequence_index = sequence_index;
        self.kv_index = sequence_index;
        self.filling_length = filling_length;
        self.phase = Phase::Prefill;
        self.notify.notify_one();
        Ok(())
    }

    pub fn transition_to_decode(&mut self) -> Result<(), TransitionError> {
        if self.phase == Phase::Decode {
            return Err(TransitionError::AlreadyInTargetState);
        }

        if self.phase != Phase::Prefill {
            return Err(TransitionError::InvalidTransition);
        }

        self.phase = Phase::Decode;
        self.filling_length = 0;
        self.notify.notify_one();
        Ok(())
    }

    pub fn transition_to_eos(&mut self) -> Result<(), TransitionError> {
        if self.phase == Phase::Eos {
            return Err(TransitionError::AlreadyInTargetState);
        }

        if !matches!(self.phase, Phase::Decode | Phase::Prefill) {
            return Err(TransitionError::InvalidTransition);
        }

        self.phase = Phase::Eos;
        self.notify.notify_one();
        Ok(())
    }

    pub fn transition_to_timeout(&mut self) -> Result<(), TransitionError> {
        if self.phase == Phase::Timeout {
            return Err(TransitionError::AlreadyInTargetState);
        }

        if !matches!(self.phase, Phase::Decode | Phase::Prefill) {
            return Err(TransitionError::InvalidTransition);
        }

        self.phase = Phase::Timeout;
        self.notify.notify_one();
        Ok(())
    }

    pub fn reset_to_start(&mut self) {
        self.sequence_index = usize::MAX;
        self.kv_index = usize::MAX;
        self.filling_length = 0;
        self.phase = Phase::Start;
        self.session_id = None;
        self.token_count = 0;
    }

    pub fn advance_sequence(&mut self, steps: usize) -> Option<Phase> {
        let previous_phase = self.phase;
        if self.phase == Phase::Eos {
            return None;
        }
        self.sequence_index += steps;

        if self.phase == Phase::Prefill {
            self.filling_length = self.filling_length.saturating_sub(steps);
            if self.filling_length == 0 {
                self.phase = Phase::Decode;
                self.notify.notify_one();
                return Some(Phase::Decode);
            }
        }

        if previous_phase != self.phase {
            Some(self.phase)
        } else {
            None
        }
    }

    pub fn can_transition(from: Phase, to: Phase) -> bool {
        match (from, to) {
            (Phase::Start, Phase::Prefill) => true,
            (Phase::Eos, Phase::Prefill) => true,
            (Phase::Timeout, Phase::Prefill) => true,
            (Phase::Prefill, Phase::Decode) => true,
            (Phase::Decode, Phase::Eos) => true,
            (Phase::Prefill, Phase::Eos) => true,
            (Phase::Decode, Phase::Timeout) => true,
            (Phase::Prefill, Phase::Timeout) => true,
            _ => false,
        }
    }
}

impl Default for SlotState {
    fn default() -> Self {
        Self::new_start_state()
    }
}
