pub mod batch_sequence;
pub mod io;
pub mod runner;
pub mod scheduling;
pub mod spin_barrier;

pub use crate::config::generation_config;
pub use crate::config::huggingface_config;

pub use crate::tensor;

pub use crate::transformer::config::Config;
pub use generation_config::GenerationConfig;
pub use huggingface_config::HfConfig;

pub use batch_sequence::BatchSequence;
pub use io::load_tiktoken;
pub use io::ChatTemplate;
pub use io::SafeTensorsLoader;
pub use runner::ServingRunner;
pub use scheduling::BatchScheduler;
pub use scheduling::{Phase, SequenceState};

/// Compatibility alias matching sample's Runner name.
pub use runner::ServingRunner as Runner;

/// Compatibility modules for existing imports
pub mod chat_template {
    pub use super::io::ChatTemplate;
}

pub mod tokenizer_loader {
    pub use super::io::load_tiktoken;
}

pub mod model_loader {
    pub use super::io::SafeTensorsLoader;
}

pub mod sequence_slice {
    pub use super::scheduling::sequence_slice::*;
}

#[cfg(test)]
mod tests {
    use super::{BatchScheduler, Phase, SequenceState, ServingRunner};
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
        let scheduler = BatchScheduler::new(8, 2, 1);
        let runner = ServingRunner::<f32>::new(Vec::new(), scheduler);

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
