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
#[repr(u8)] // 优化: 显式指定为 u8，确保只占 1 字节
/// ```mermaid
/// stateDiagram-v2
///     [*] --> Start
///     Start --> Prefill: assign_slot
///     Prefill --> Decode: prefill_done
///     Decode --> Eos: generation_done
///
///     Eos --> Prefill: same_user_continue
///     Eos --> Timeout: enter_timeout_state
///
///     Timeout --> Prefill: resume_same_user
///     Timeout --> Start: recycle_for_new_user
///     Timeout --> Closed: terminate_slot
///     Closed --> [*]
/// ```
pub enum Phase {
    Start,
    Prefill,
    Decode,
    Timeout,
    Eos,
}
