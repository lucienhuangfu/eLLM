#[derive(Clone)]
pub struct TokenRecord {
    pub token_id: usize,
    pub batch_index: usize,
    pub position_index: usize,
}

pub struct UserRecord {
    pub sequence_index: usize,
    pub snapshot_sequence_index: usize,
    pub kv_index: usize,
    pub phase: Phase,
}

pub struct LastPrefillRecord {
    pub prefill_index: usize,
    pub decode_index: usize,
}

pub struct TokenRecordList {
    pub records: Vec<TokenRecord>,
    pub current_size: usize,
    pub max_size: usize,
}

pub struct LastPrefillList {
    pub records: Vec<LastPrefillRecord>,
    pub current_size: usize,
    pub max_size: usize,
}

pub struct UserList {
    pub records: Vec<UserRecord>,
    pub current_size: usize,
    pub max_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill_begin,
    Prefill_end,
    Decode,
    Eos,
}

/*
impl TokenRecord {
    /// Creates a raw pointer to an array of TokenRecords from a Vec.
    /// The caller is responsible for freeing the memory.
    pub fn new_raw_array(records: Vec<TokenRecord>) -> *mut TokenRecord {
        let mut boxed_slice = records.into_boxed_slice();
        let ptr = boxed_slice.as_mut_ptr();
        std::mem::forget(boxed_slice); // Prevent deallocation
        ptr
    }

    /// Access a record at a specific index from a raw pointer.
    /// # Safety
    /// The caller must ensure the pointer is valid and the index is within bounds.
    pub unsafe fn get_from_raw<'a>(ptr: *mut TokenRecord, index: usize) -> &'a mut TokenRecord {
        &mut *ptr.add(index)
    }

    /// Access a specific field (e.g., sequence_index) from a raw pointer at an index.
    /// # Safety
    /// The caller must ensure the pointer is valid and the index is within bounds.
    pub unsafe fn get_sequence_index_from_raw(ptr: *mut TokenRecord, index: usize) -> usize {
        (*ptr.add(index)).sequence_index
    }
} */
