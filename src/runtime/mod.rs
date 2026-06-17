pub mod error;
pub mod executor;
pub mod io;
pub mod scheduler;
pub mod session;
pub mod state;

pub use crate::config::generation_config::GenerationConfig;
pub use crate::config::huggingface_config::HfConfig;
pub use crate::tensor;
pub use crate::transformer::config::Config;

pub use executor::{ServingRunner, SpinBarrier};
pub use io::{load_tiktoken, ChatTemplate, SafeTensorsLoader};
pub use scheduler::{
    BatchPlan, DefaultSchedulerStrategy, PrefillCandidate, ScheduleTask, Scheduler,
    SchedulerStrategy,
};
pub use session::{SessionHandle, SessionManager, SessionMode, SlotAllocator};
pub use state::{
    BatchSequence, DecodeList, DecodeLookupResult, Phase, SequenceSlice, SequenceState,
    SequenceStateMachine, TransitionError, build_batch_sequence, build_sequence_state,
};

pub use executor::runner::ServingRunner as Runner;

#[cfg(test)]
mod tests {
    use super::{Phase, Scheduler, SequenceState, ServingRunner};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::broadcast;
    use tokio::sync::Notify;

    #[test]
    fn runtime_reexports_are_constructible() {
        use crate::operators::send_sync_ptr::SharedMut;

        let prefill_state = SequenceState::new_prefill_state(8, 4);
        let decode_state = SequenceState::new_decode_state(16, 16);

        let (sender, _) = broadcast::channel(4);
        let batch_list = Arc::new(SharedMut::new(Vec::new()));
        let scheduler = Scheduler::new(
            8,
            2,
            1,
            1,
            Duration::from_millis(100),
            sender.clone(),
            batch_list,
        );
        let runner =
            ServingRunner::<f32>::new(Vec::new(), Arc::clone(&scheduler.batch_list()), sender);

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

        let _ = runner;
    }
}
