use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;

const LRU_SENTINEL: usize = usize::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Phase {
    Start,
    Prefill,
    Decode,
    Timeout,
    Eos,
}

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

// ── Build helpers ──────────────────────────────────────────

pub fn build_slot_state(batch_size: usize) -> Vec<SlotState> {
    (0..batch_size)
        .map(|_| SlotState::new_start_state())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_phase_repr_and_ordering() {
        assert_eq!(std::mem::size_of::<Phase>(), 1);
        assert!((Phase::Start as u8) < (Phase::Prefill as u8));
        assert!((Phase::Prefill as u8) < (Phase::Decode as u8));
        assert!((Phase::Decode as u8) < (Phase::Timeout as u8));
        assert!((Phase::Timeout as u8) < (Phase::Eos as u8));
    }

    #[test]
    fn test_slot_state_lifecycle() {
        let mut state = SlotState::new_start_state();
        assert_eq!(state.phase, Phase::Start);
        assert!(!state.is_active());
        assert!(state.is_available());

        state.transition_to_prefill(100, 50).unwrap();
        assert_eq!(state.phase, Phase::Prefill);
        assert!(state.is_active());
        assert_eq!(state.sequence_index, 100);
        assert_eq!(state.filling_length, 50);

        let change = state.advance_sequence(50);
        assert_eq!(change, Some(Phase::Decode));
        assert_eq!(state.phase, Phase::Decode);
        assert_eq!(state.sequence_index, 150);

        state.transition_to_eos().unwrap();
        assert!(!state.is_active());
        assert!(state.is_available());
    }

    #[test]
    fn test_can_transition_all_valid_paths() {
        assert!(SlotState::can_transition(Phase::Start, Phase::Prefill));
        assert!(SlotState::can_transition(Phase::Eos, Phase::Prefill));
        assert!(SlotState::can_transition(Phase::Timeout, Phase::Prefill));
        assert!(SlotState::can_transition(Phase::Prefill, Phase::Decode));
        assert!(SlotState::can_transition(Phase::Decode, Phase::Eos));
        assert!(SlotState::can_transition(Phase::Prefill, Phase::Eos));
        assert!(SlotState::can_transition(Phase::Decode, Phase::Timeout));
        assert!(SlotState::can_transition(Phase::Prefill, Phase::Timeout));

        assert!(!SlotState::can_transition(Phase::Start, Phase::Decode));
        assert!(!SlotState::can_transition(Phase::Decode, Phase::Prefill));
        assert!(!SlotState::can_transition(Phase::Eos, Phase::Decode));
    }

    #[test]
    fn test_invalid_transitions() {
        let mut state = SlotState::new_start_state();
        assert!(state.transition_to_decode().is_err());
        assert!(state.transition_to_eos().is_err());
    }

    #[test]
    fn test_advance_sequence_partial_and_saturation() {
        let mut state = SlotState::new_prefill_state(0, 10);
        assert!(state.advance_sequence(3).is_none());
        assert_eq!(state.filling_length, 7);
        assert_eq!(state.sequence_index, 3);

        assert_eq!(state.advance_sequence(100), Some(Phase::Decode));
        assert_eq!(state.filling_length, 0);
    }

    #[test]
    fn test_touch_updates_timestamp() {
        let mut state = SlotState::new_decode_state(0, 0);
        let original = state.last_accessed;
        std::thread::sleep(Duration::from_millis(1));
        state.touch();
        assert!(state.last_accessed > original);
    }

    #[test]
    fn test_reset_to_start() {
        let mut state = SlotState::new_decode_state(5, 5);
        state.session_id = Some("s1".into());
        state.token_count = 10;
        state.reset_to_start();
        assert_eq!(state.phase, Phase::Start);
        assert_eq!(state.sequence_index, usize::MAX);
        assert!(state.session_id.is_none());
        assert_eq!(state.token_count, 0);
    }
}
