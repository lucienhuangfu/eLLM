mod io;
pub mod runner;
mod scheduling;

pub use crate::tensor;

pub use crate::config::{
    ChatArgs, ChatConfig, Cli, CliCommand, Command, Config, ConfigError, GenerationConfig,
    HfConfig, ModelConfig, ModelDtype, ResolvedConfig, ResolvedModelConfig, SchedulerConfig,
    SchedulingPolicy, ServeArgs, ServeConfig, SharedArgs,
};

pub use io::{load_tiktoken, BatchSequence, ChatTemplate, FromSafetensors, SafeTensorsLoader};
pub use runner::Runner;
pub use scheduling::{
    BatchScheduleStep, BatchScheduler, DecodeList, DecodeLookupResult, Phase, SchedulingMode,
    SequenceSlice, SequenceState,
};

#[cfg(test)]
mod tests {
    use super::{BatchScheduler, Phase, Runner, SequenceState};
    use std::sync::Arc;
    use tokio::sync::Notify;

    #[test]
    fn runtime_reexports_are_constructible() {
        let prefill_state = SequenceState {
            sequence_index: 8,
            kv_index: 12,
            filling_length: 4,
            phase: Phase::Prefill,
            notify: Arc::new(Notify::new()),
        };
        let decode_state = SequenceState {
            sequence_index: 16,
            kv_index: 16,
            filling_length: 0,
            phase: Phase::Decode,
            notify: Arc::new(Notify::new()),
        };
        let scheduler = BatchScheduler::new(8, 2, 8, 1);
        let runner = Runner::<f32>::new(Vec::new(), scheduler);

        assert_eq!(prefill_state.sequence_index, 8);
        assert_eq!(prefill_state.kv_index, 12);
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
