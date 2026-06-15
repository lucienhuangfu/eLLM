use std::fmt;

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