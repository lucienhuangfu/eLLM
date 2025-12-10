#[derive(Clone)]
pub struct TokenRecord {
    // pub token_id: usize,
    // 优化: 使用 usize (4 bytes) 代替 usize (8 bytes)。
    // 内存占用从 16 bytes -> 8 bytes。
    pub batch_index: usize,
    pub position_index: usize,
}

pub struct BatchRecord {
    // 优化: 使用 usize
    pub sequence_index: usize,
    // pub snapshot_sequence_index: usize,
    pub kv_index: usize,
    pub phase: Phase,
    // 内存布局优化:
    // 原来: 8 + 8 + 1 (+7 padding) = 24 bytes
    // 现在: 4 + 4 + 1 (+3 padding) = 12 bytes
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillEndRecord {
    // 优化: 使用 usize
    pub prefill_end_index: usize,
    pub lift_index: usize,
}

pub struct TokenList {
    pub token_records: Box<[TokenRecord]>,
    pub current_token_size: usize, // 保留，表示有效长度
    pub lift_records: Box<[PrefillEndRecord]>,
    pub current_lift_size: usize, // 保留，表示有效长度
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
