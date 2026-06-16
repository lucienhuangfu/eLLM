use std::sync::Arc;
use tokio::sync::Notify;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    Start,
    Prefill,
    Decode,
    Timeout,
    Eos,
}

#[derive(Clone)]
pub struct SequenceState {
    pub sequence_index: usize,
    pub kv_index: usize,
    pub filling_length: usize,
    pub phase: Phase,
    pub(crate) notify: Arc<Notify>,
}

impl SequenceState {
    pub fn new_start_state() -> Self {
        Self {
            sequence_index: usize::MAX,
            kv_index: usize::MAX,
            filling_length: 0,
            phase: Phase::Start,
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn new_prefill_state(sequence_index: usize, filling_length: usize) -> Self {
        Self {
            sequence_index,
            kv_index: sequence_index,
            filling_length,
            phase: Phase::Prefill,
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn new_decode_state(sequence_index: usize, kv_index: usize) -> Self {
        Self {
            sequence_index,
            kv_index,
            filling_length: 0,
            phase: Phase::Decode,
            notify: Arc::new(Notify::new()),
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
}

impl Default for SequenceState {
    fn default() -> Self {
        Self::new_start_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_enum_variants() {
        let phases = [
            Phase::Start,
            Phase::Prefill,
            Phase::Decode,
            Phase::Timeout,
            Phase::Eos,
        ];

        for (i, phase1) in phases.iter().enumerate() {
            for (j, phase2) in phases.iter().enumerate() {
                if i == j {
                    assert_eq!(phase1, phase2, "Same phase should be equal");
                } else {
                    assert_ne!(phase1, phase2, "Different phases should not be equal");
                }
            }
        }
    }

    #[test]
    fn test_new_start_state() {
        let state = SequenceState::new_start_state();

        assert_eq!(
            state.sequence_index,
            usize::MAX,
            "sequence_index should be MAX"
        );
        assert_eq!(state.kv_index, usize::MAX, "kv_index should be MAX");
        assert_eq!(state.filling_length, 0, "filling_length should be 0");
        assert_eq!(state.phase, Phase::Start, "phase should be Start");
    }

    #[test]
    fn test_new_prefill_state() {
        let state = SequenceState::new_prefill_state(5, 10);

        assert_eq!(state.sequence_index, 5, "sequence_index should be 5");
        assert_eq!(
            state.kv_index, 5,
            "kv_index should equal sequence_index in prefill"
        );
        assert_eq!(state.filling_length, 10, "filling_length should be 10");
        assert_eq!(state.phase, Phase::Prefill, "phase should be Prefill");
    }

    #[test]
    fn test_new_decode_state() {
        let state = SequenceState::new_decode_state(10, 3);

        assert_eq!(state.sequence_index, 10, "sequence_index should be 10");
        assert_eq!(state.kv_index, 3, "kv_index should be 3");
        assert_eq!(state.filling_length, 0, "filling_length should be 0");
        assert_eq!(state.phase, Phase::Decode, "phase should be Decode");
    }

    #[test]
    fn test_transition_to_decode() {
        let mut state = SequenceState::new_prefill_state(0, 5);

        assert_eq!(state.phase, Phase::Prefill);

        state.transition_to_decode();

        assert_eq!(
            state.phase,
            Phase::Decode,
            "phase should transition to Decode"
        );
        assert_eq!(
            state.filling_length, 0,
            "filling_length should be reset to 0"
        );
    }

    #[test]
    fn test_transition_to_eos() {
        let mut state = SequenceState::new_decode_state(10, 5);

        state.transition_to_eos();

        assert_eq!(state.phase, Phase::Eos, "phase should transition to Eos");
    }

    #[test]
    fn test_transition_to_timeout() {
        let mut state = SequenceState::new_decode_state(10, 5);

        state.transition_to_timeout();

        assert_eq!(
            state.phase,
            Phase::Timeout,
            "phase should transition to Timeout"
        );
    }

    #[test]
    fn test_reset_to_start() {
        let mut state = SequenceState::new_decode_state(10, 5);

        state.reset_to_start();

        assert_eq!(
            state.sequence_index,
            usize::MAX,
            "sequence_index should be reset"
        );
        assert_eq!(state.kv_index, usize::MAX, "kv_index should be reset");
        assert_eq!(state.filling_length, 0, "filling_length should be reset");
        assert_eq!(state.phase, Phase::Start, "phase should be reset to Start");
    }

    #[test]
    fn test_is_active() {
        let start_state = SequenceState::new_start_state();
        assert!(!start_state.is_active(), "Start state should not be active");

        let prefill_state = SequenceState::new_prefill_state(0, 5);
        assert!(prefill_state.is_active(), "Prefill state should be active");

        let decode_state = SequenceState::new_decode_state(10, 5);
        assert!(decode_state.is_active(), "Decode state should be active");

        let mut eos_state = SequenceState::new_decode_state(10, 5);
        eos_state.transition_to_eos();
        assert!(!eos_state.is_active(), "Eos state should not be active");

        let mut timeout_state = SequenceState::new_decode_state(10, 5);
        timeout_state.transition_to_timeout();
        assert!(
            !timeout_state.is_active(),
            "Timeout state should not be active"
        );
    }

    #[test]
    fn test_is_available() {
        let start_state = SequenceState::new_start_state();
        assert!(
            start_state.is_available(),
            "Start state should be available"
        );

        let prefill_state = SequenceState::new_prefill_state(0, 5);
        assert!(
            !prefill_state.is_available(),
            "Prefill state should not be available"
        );

        let decode_state = SequenceState::new_decode_state(10, 5);
        assert!(
            !decode_state.is_available(),
            "Decode state should not be available"
        );

        let mut eos_state = SequenceState::new_decode_state(10, 5);
        eos_state.transition_to_eos();
        assert!(eos_state.is_available(), "Eos state should be available");

        let mut timeout_state = SequenceState::new_decode_state(10, 5);
        timeout_state.transition_to_timeout();
        assert!(
            !timeout_state.is_available(),
            "Timeout state should not be available"
        );
    }

    #[test]
    fn test_advance_sequence_partial() {
        let mut state = SequenceState::new_prefill_state(0, 5);

        state.advance_sequence(2);

        assert_eq!(
            state.sequence_index, 2,
            "sequence_index should advance by 2"
        );
        assert_eq!(
            state.filling_length, 3,
            "filling_length should decrease by 2"
        );
        assert_eq!(state.phase, Phase::Prefill, "phase should remain Prefill");
    }

    #[test]
    fn test_advance_sequence_full() {
        let mut state = SequenceState::new_prefill_state(0, 3);

        state.advance_sequence(3);

        assert_eq!(
            state.sequence_index, 3,
            "sequence_index should advance by 3"
        );
        assert_eq!(state.filling_length, 0, "filling_length should be 0");
        assert_eq!(
            state.phase,
            Phase::Decode,
            "phase should transition to Decode"
        );
    }

    #[test]
    fn test_advance_sequence_in_decode() {
        let mut state = SequenceState::new_decode_state(10, 5);

        state.advance_sequence(1);

        assert_eq!(state.sequence_index, 11, "sequence_index should advance");
        assert_eq!(state.phase, Phase::Decode, "phase should remain Decode");
        assert_eq!(state.filling_length, 0, "filling_length should remain 0");
    }

    #[test]
    fn test_default() {
        let state: SequenceState = Default::default();

        assert_eq!(state.phase, Phase::Start);
        assert_eq!(state.sequence_index, usize::MAX);
        assert_eq!(state.kv_index, usize::MAX);
        assert_eq!(state.filling_length, 0);
    }
}
