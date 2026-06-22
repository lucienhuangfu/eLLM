pub mod error;
pub mod executor;
pub mod io;
pub mod plan;
pub mod scheduler;
pub mod session;
pub mod state;

#[cfg(test)]
pub mod integration_test;

pub use crate::config::generation_config::GenerationConfig;
pub use crate::config::huggingface_config::HfConfig;
pub use crate::tensor;
pub use crate::transformer::config::Config;

pub use executor::ExecutorPool;
pub use io::{load_tiktoken, ChatTemplate, SafeTensorsLoader};
pub use scheduler::{ScheduleTask, Scheduler};
pub use session::{SessionHandle, SessionMode, SlotManager};
pub use state::{
    build_batch_sequence, build_slot_state, BatchSequence, DecodeList, DecodeLookupResult,
    Phase, SequenceSlice, SharedState, SlotState, SlotStateMachine, TransitionError,
};

#[cfg(test)]
mod tests {
    use super::{Phase, SlotState};
    use std::sync::Arc;

    #[test]
    fn runtime_reexports_are_constructible() {
        use crate::operators::send_sync_ptr::SharedMut;

        let prefill_state = SlotState::new_prefill_state(8, 4);
        let decode_state = SlotState::new_decode_state(16, 16);

        assert_eq!(prefill_state.sequence_index, 8);
        assert_eq!(prefill_state.kv_index, 8);
        assert_eq!(prefill_state.filling_length, 4);
        assert_eq!(prefill_state.phase, Phase::Prefill);
        assert_eq!(Arc::strong_count(&prefill_state.notify), 1);

        assert_eq!(decode_state.sequence_index, 16);
        assert_eq!(decode_state.kv_index, 16);
        assert_eq!(decode_state.filling_length, 0);
        assert_eq!(decode_state.phase, Phase::Decode);
        assert_eq!(Arc::strong_count(&decode_state.notify), 1);
    }
}