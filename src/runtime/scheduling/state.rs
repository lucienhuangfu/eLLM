use std::sync::Arc;
use tokio::sync::Notify;

pub struct SequenceState {
    pub sequence_index: usize,
    pub kv_index: usize,
    pub filling_length: usize,
    pub phase: Phase,
    pub notify: Arc<Notify>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    Start,
    Prefill,
    Decode,
    Timeout,
    Eos,
}
