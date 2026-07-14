use std::fmt;

// ── SlotError ──────────────────────────────────────────────

#[derive(Debug)]
pub enum SlotError {
    AllocatorUnavailable,
    SlotQueueEmpty,
    SlotNotFound,
}

impl fmt::Display for SlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlotError::AllocatorUnavailable => write!(f, "Slot allocator unavailable"),
            SlotError::SlotQueueEmpty => write!(f, "Slot queue empty while permit acquired"),
            SlotError::SlotNotFound => write!(f, "Slot not found"),
        }
    }
}

impl std::error::Error for SlotError {}

pub type SlotResult<T> = Result<T, SlotError>;

// ── SessionMode ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionMode {
    Reusable,
    NonReusable,
}

// ── SessionHandle ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SessionHandle {
    pub session_id: String,
    pub slot_index: usize,
    pub is_reused: bool,
}

impl SessionHandle {
    pub fn new(session_id: String, slot_index: usize) -> Self {
        Self {
            session_id,
            slot_index,
            is_reused: false,
        }
    }

    pub fn reused(session_id: String, slot_index: usize) -> Self {
        Self {
            session_id,
            slot_index,
            is_reused: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_error_display() {
        assert_eq!(
            SlotError::AllocatorUnavailable.to_string(),
            "Slot allocator unavailable"
        );
        assert_eq!(
            SlotError::SlotQueueEmpty.to_string(),
            "Slot queue empty while permit acquired"
        );
        assert_eq!(SlotError::SlotNotFound.to_string(), "Slot not found");
    }

    #[test]
    fn test_slot_error_propagation() {
        fn inner() -> SlotResult<()> {
            Err(SlotError::SlotNotFound)
        }
        fn outer() -> SlotResult<i32> {
            inner().map(|_| 42)
        }
        assert!(matches!(outer(), Err(SlotError::SlotNotFound)));
    }
}
