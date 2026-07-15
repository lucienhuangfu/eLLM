pub mod slot_manager;
pub mod slot_state;
pub mod types;

pub use slot_manager::SlotManager;
pub use slot_state::{Phase, SlotState, TransitionError};
pub use types::{SessionHandle, SessionMode, SlotError, SlotResult};
