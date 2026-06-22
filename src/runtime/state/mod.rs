pub mod batch;
pub mod core;
pub mod machine;
pub mod sequence;
pub mod shared;
pub mod state_init;
pub mod types;

pub use batch::BatchSequence;
pub use core::SlotState;
pub use machine::{SlotStateMachine, TransitionError};
pub use sequence::{DecodeList, DecodeLookupResult, SequenceSlice};
pub use shared::SharedState;
pub use state_init::{build_batch_sequence, build_slot_state};
pub use types::Phase;