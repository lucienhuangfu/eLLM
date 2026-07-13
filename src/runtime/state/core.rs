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
    /// Private helper to create a slot with common defaults.
    fn fresh(phase: Phase, sequence_index: usize, kv_index: usize, filling_length: usize) -> Self {
        let now = Instant::now();
        Self {
            sequence_index,
            kv_index,
            filling_length,
            phase,
            session_id: None,
            token_count: 0,
            created_at: now,
            last_accessed: now,
            notify: Arc::new(Notify::new()),
            lru_prev: LRU_SENTINEL,
            lru_next: LRU_SENTINEL,
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

    /// Core transition validator + phase setter. Reuses [`can_transition`] so the
    /// allowed-from set lives in exactly one place.
    fn transition(&mut self, target: Phase) -> Result<(), TransitionError> {
        if self.phase == target {
            return Err(TransitionError::AlreadyInTargetState);
        }
        if !Self::can_transition(self.phase, target) {
            return Err(TransitionError::InvalidTransition);
        }
        self.phase = target;
        self.notify.notify_one();
        Ok(())
    }

    pub fn transition_to_prefill(
        &mut self,
        sequence_index: usize,
        filling_length: usize,
    ) -> Result<(), TransitionError> {
        self.transition(Phase::Prefill)?;
        self.sequence_index = sequence_index;
        self.kv_index = sequence_index;
        self.filling_length = filling_length;
        Ok(())
    }

    pub fn transition_to_decode(&mut self) -> Result<(), TransitionError> {
        self.transition(Phase::Decode)?;
        self.filling_length = 0;
        Ok(())
    }

    pub fn transition_to_eos(&mut self) -> Result<(), TransitionError> {
        self.transition(Phase::Eos)
    }

    pub fn transition_to_timeout(&mut self) -> Result<(), TransitionError> {
        self.transition(Phase::Timeout)
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
        matches!(
            (from, to),
            (Phase::Start, Phase::Prefill)
                | (Phase::Eos, Phase::Prefill)
                | (Phase::Timeout, Phase::Prefill)
                | (Phase::Prefill, Phase::Decode)
                | (Phase::Decode, Phase::Eos)
                | (Phase::Prefill, Phase::Eos)
                | (Phase::Decode, Phase::Timeout)
                | (Phase::Prefill, Phase::Timeout)
        )
    }
}

impl Default for SlotState {
    fn default() -> Self {
        Self::new_start_state()
    }
}
