use super::types::{Phase, SequenceState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionError {
    InvalidTransition,
    AlreadyInTargetState,
}

pub struct SequenceStateMachine;

impl SequenceStateMachine {
    pub fn new_start_state() -> SequenceState {
        SequenceState::new_start_state()
    }

    pub fn new_prefill_state(sequence_index: usize, filling_length: usize) -> SequenceState {
        SequenceState::new_prefill_state(sequence_index, filling_length)
    }

    pub fn new_decode_state(sequence_index: usize, kv_index: usize) -> SequenceState {
        SequenceState::new_decode_state(sequence_index, kv_index)
    }

    pub fn transition_to_prefill(
        state: &mut SequenceState,
        sequence_index: usize,
        filling_length: usize,
    ) -> Result<(), TransitionError> {
        if state.phase == Phase::Prefill {
            return Err(TransitionError::AlreadyInTargetState);
        }

        if !matches!(state.phase, Phase::Start | Phase::Eos | Phase::Timeout) {
            return Err(TransitionError::InvalidTransition);
        }

        state.sequence_index = sequence_index;
        state.kv_index = sequence_index;
        state.filling_length = filling_length;
        state.phase = Phase::Prefill;
        state.notify.notify_one();
        Ok(())
    }

    pub fn transition_to_decode(state: &mut SequenceState) -> Result<(), TransitionError> {
        if state.phase == Phase::Decode {
            return Err(TransitionError::AlreadyInTargetState);
        }

        if state.phase != Phase::Prefill {
            return Err(TransitionError::InvalidTransition);
        }

        state.phase = Phase::Decode;
        state.filling_length = 0;
        state.notify.notify_one();
        Ok(())
    }

    pub fn transition_to_eos(state: &mut SequenceState) -> Result<(), TransitionError> {
        if state.phase == Phase::Eos {
            return Err(TransitionError::AlreadyInTargetState);
        }

        if !matches!(state.phase, Phase::Decode | Phase::Prefill) {
            return Err(TransitionError::InvalidTransition);
        }

        state.phase = Phase::Eos;
        state.notify.notify_one();
        Ok(())
    }

    pub fn transition_to_timeout(state: &mut SequenceState) -> Result<(), TransitionError> {
        if state.phase == Phase::Timeout {
            return Err(TransitionError::AlreadyInTargetState);
        }

        if !matches!(state.phase, Phase::Decode | Phase::Prefill) {
            return Err(TransitionError::InvalidTransition);
        }

        state.phase = Phase::Timeout;
        state.notify.notify_one();
        Ok(())
    }

    pub fn reset_to_start(state: &mut SequenceState) {
        state.sequence_index = usize::MAX;
        state.kv_index = usize::MAX;
        state.filling_length = 0;
        state.phase = Phase::Start;
    }

    pub fn advance_sequence(state: &mut SequenceState, steps: usize) -> Option<Phase> {
        let previous_phase = state.phase;
        state.sequence_index += steps;

        if state.phase == Phase::Prefill {
            state.filling_length = state.filling_length.saturating_sub(steps);
            if state.filling_length == 0 {
                let _ = Self::transition_to_decode(state);
                return Some(Phase::Decode);
            }
        }

        if previous_phase != state.phase {
            Some(state.phase)
        } else {
            None
        }
    }

    pub fn is_active(state: &SequenceState) -> bool {
        matches!(state.phase, Phase::Prefill | Phase::Decode)
    }

    pub fn is_available(state: &SequenceState) -> bool {
        matches!(state.phase, Phase::Start | Phase::Eos)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_from_start_to_prefill() {
        let mut state = SequenceStateMachine::new_start_state();
        assert_eq!(state.phase, Phase::Start);

        let result = SequenceStateMachine::transition_to_prefill(&mut state, 0, 10);
        assert!(result.is_ok());
        assert_eq!(state.phase, Phase::Prefill);
        assert_eq!(state.sequence_index, 0);
        assert_eq!(state.filling_length, 10);
    }

    #[test]
    fn transition_from_prefill_to_decode() {
        let mut state = SequenceStateMachine::new_prefill_state(0, 5);
        assert_eq!(state.phase, Phase::Prefill);

        let result = SequenceStateMachine::transition_to_decode(&mut state);
        assert!(result.is_ok());
        assert_eq!(state.phase, Phase::Decode);
        assert_eq!(state.filling_length, 0);
    }

    #[test]
    fn invalid_transition_returns_error() {
        let mut state = SequenceStateMachine::new_start_state();
        let result = SequenceStateMachine::transition_to_decode(&mut state);
        assert!(matches!(result, Err(TransitionError::InvalidTransition)));
    }

    #[test]
    fn advance_sequence_automatically_transitions_to_decode() {
        let mut state = SequenceStateMachine::new_prefill_state(0, 3);
        assert_eq!(state.phase, Phase::Prefill);

        let phase_change = SequenceStateMachine::advance_sequence(&mut state, 3);
        assert_eq!(phase_change, Some(Phase::Decode));
        assert_eq!(state.phase, Phase::Decode);
        assert_eq!(state.sequence_index, 3);
    }

    #[test]
    fn advance_sequence_partial() {
        let mut state = SequenceStateMachine::new_prefill_state(0, 5);
        assert_eq!(state.phase, Phase::Prefill);

        let phase_change = SequenceStateMachine::advance_sequence(&mut state, 2);
        assert_eq!(phase_change, None);
        assert_eq!(state.phase, Phase::Prefill);
        assert_eq!(state.sequence_index, 2);
        assert_eq!(state.filling_length, 3);
    }

    #[test]
    fn can_transition_validates_transitions() {
        assert!(SequenceStateMachine::can_transition(
            Phase::Start,
            Phase::Prefill
        ));
        assert!(SequenceStateMachine::can_transition(
            Phase::Prefill,
            Phase::Decode
        ));
        assert!(SequenceStateMachine::can_transition(
            Phase::Decode,
            Phase::Eos
        ));
        assert!(!SequenceStateMachine::can_transition(
            Phase::Decode,
            Phase::Prefill
        ));
        assert!(!SequenceStateMachine::can_transition(
            Phase::Eos,
            Phase::Decode
        ));
    }

    #[test]
    fn reset_to_start_clears_state() {
        let mut state = SequenceStateMachine::new_decode_state(10, 5);
        assert_eq!(state.phase, Phase::Decode);
        assert_eq!(state.sequence_index, 10);

        SequenceStateMachine::reset_to_start(&mut state);
        assert_eq!(state.phase, Phase::Start);
        assert_eq!(state.sequence_index, usize::MAX);
        assert_eq!(state.kv_index, usize::MAX);
        assert_eq!(state.filling_length, 0);
    }

    #[test]
    fn transition_to_eos_from_decode() {
        let mut state = SequenceStateMachine::new_decode_state(10, 5);
        let result = SequenceStateMachine::transition_to_eos(&mut state);
        assert!(result.is_ok());
        assert_eq!(state.phase, Phase::Eos);
    }

    #[test]
    fn transition_to_timeout_from_decode() {
        let mut state = SequenceStateMachine::new_decode_state(10, 5);
        let result = SequenceStateMachine::transition_to_timeout(&mut state);
        assert!(result.is_ok());
        assert_eq!(state.phase, Phase::Timeout);
    }
}
