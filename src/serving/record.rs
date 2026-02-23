use std::sync::Arc;
use tokio::sync::Notify;

#[derive(Clone)]
pub struct SequenceSlice {
    pub batch_index: usize,
    pub sequence_index: usize,
    pub token_start_index: usize,
    pub lift_index: usize,
    pub length: usize,
}

pub struct SequenceState {
    pub sequence_index: usize,
    pub kv_index: usize,
    pub phase: Phase,
    pub notify: Arc<Notify>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)] // 优化: 显式指定为 u8，确保只占 1 字节
pub enum Phase {
    Prefill,
    Decode,
    Eos,
}
