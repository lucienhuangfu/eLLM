use std::sync::Arc;
use tokio::sync::Notify;

#[derive(Clone)]
pub struct SequenceSlice {
    pub batch_index: usize,
    pub sequence_index: usize,
    pub token_start_index: usize,
    pub length: usize,
}

pub struct ThreadTask {
    pub slices: Box<[SequenceSlice]>,
    pub current_size: usize, // 保留，表示有效长度
}

pub struct TaskList {
    pub tasks: Box<[ThreadTask]>,
    pub current_size: usize, // 保留，表示有效长度
    pub max_token_size: usize,

}



pub struct BatchRecord {
    pub sequence_index: usize,
    pub snapshot_sequence_index: usize,
    pub kv_index: usize,
    pub phase: Phase,
    pub prompt_length: usize,
    pub notify: Arc<Notify>,
}

pub struct BatchList {
    pub records: Box<[BatchRecord]>,
    pub current_size: usize, // 保留，表示有效长度
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)] // 优化: 显式指定为 u8，确保只占 1 字节
pub enum Phase {
    PrefillBegin,
    PrefillEnd,
    Decode,
    Eos,
}

/*
#[derive(Clone, Copy, Debug)]
pub struct PrefillEndRecord {
    pub batch_index: usize,
    pub sequence_index: usize,
    pub token_start_index: usize,
    // pub prefill_end_index: usize,
    // pub lift_index: usize,
}

pub struct PrefillEndRecordList {
    pub lift_records: Box<[PrefillEndRecord]>,
    pub current_size: usize, // 保留，表示有效长度
} 
*/
