pub mod allocator;
pub mod manager;
pub mod types;

pub use allocator::SlotAllocator;
pub use manager::SessionManager;
pub use types::{DialogueSession, SessionHandle, SessionMode};
