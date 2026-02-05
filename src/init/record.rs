use std::sync::Arc;
use tokio::sync::Notify;

#[derive(Clone)]
pub struct SequenceSlice {
    pub batch_index: usize,
    pub position_index: usize,
    pub length: usize,
}

pub struct ThreadTask {
    pub slices: Box<[SequenceSlice]>,
    pub current_size: usize, // 保留，表示有效长度
}

pub struct AllTask {
    pub tasks: Box<[ThreadTask]>,
    pub current_size: usize, // 保留，表示有效长度
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillEndRecord {
    pub batch_index: usize,
    pub position_index: usize,
    pub prefill_end_index: usize,
    pub lift_index: usize,
}

pub struct PrefillEndRecordList {
    pub lift_records: Box<[PrefillEndRecord]>,
    pub current_size: usize, // 保留，表示有效长度
}


pub struct BatchRecord {
    // 优化: 使用 usize
    pub sequence_index: usize,
    // pub snapshot_sequence_index: usize,
    pub kv_index: usize,
    pub phase: Phase,
    pub prompt_length: usize,
    pub notify: Arc<Notify>,
    // 内存布局优化:
    // 原来: 8 + 8 + 1 (+7 padding) = 24 bytes
    // 现在: 4 + 4 + 1 + 8 (+ ? padding)
}








pub struct BatchList {
    pub records: Box<[BatchRecord]>,
    pub current_size: usize, // 保留，表示有效长度
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)] // 优化: 显式指定为 u8，确保只占 1 字节
pub enum Phase {
    Prefill_begin,
    Prefill_end,
    Decode,
    Eos,
}

/* 
pub struct TokenList {
    pub token_records: Box<[TokenRecord]>,
    pub current_token_size: usize, // 保留，表示有效长度

}*/