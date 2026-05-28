use std::ops::Deref;
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct SequenceSlice {
    pub token_start_index: usize,
    pub batch_index: usize,
    pub sequence_index: usize,
    pub length: usize,
    pub last_token_flag: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeLookupResult {
    pub batch_index: usize,
    pub sequence_index: usize,
    pub slice_index: usize,
}

#[derive(Clone, Default)]
pub struct DecodeList {
    slices: Vec<SequenceSlice>,
    len: usize,
}

impl DecodeList {
    pub fn with_capacity(capacity: usize) -> Self {
        let mut slices = Vec::with_capacity(capacity);
        slices.resize(capacity, SequenceSlice::default());
        Self { slices, len: 0 }
    }

    pub fn push(&mut self, slice: SequenceSlice) {
        debug_assert!(self.len <= self.slices.len());
        if self.len == self.slices.len() {
            self.slices.push(slice);
        } else {
            self.slices[self.len] = slice;
        }
        self.len += 1;
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn total_token_count(&self) -> usize {
        self.slices[..self.len]
            .iter()
            .map(|slice| slice.length)
            .sum()
    }

    pub fn lookup_global_index(&self, global_index: usize) -> Option<DecodeLookupResult> {
        let slices = self.as_slice();
        let slice_index =
            slices.partition_point(|slice| slice.token_start_index + slice.length <= global_index);
        let slice = slices.get(slice_index)?;
        if global_index < slice.token_start_index {
            return None;
        }

        Some(DecodeLookupResult {
            batch_index: slice.batch_index,
            sequence_index: slice.sequence_index + (global_index - slice.token_start_index),
            slice_index,
        })
    }

    pub fn walk_global_range(
        &self,
        global_begin: usize,
        global_end: usize,
        mut visit: impl FnMut(usize, usize, usize),
    ) {
        if global_begin >= global_end {
            return;
        }

        let Some(found) = self.lookup_global_index(global_begin) else {
            return;
        };

        let mut slice_index = found.slice_index;
        let mut global_index = global_begin;
        while global_index < global_end {
            let Some(slice) = self.slices.get(slice_index) else {
                break;
            };

            let slice_end = slice.token_start_index + slice.length;
            if global_index < slice.token_start_index {
                break;
            }

            let visit_end = global_end.min(slice_end);
            while global_index < visit_end {
                visit(
                    global_index,
                    slice.batch_index,
                    slice.sequence_index + (global_index - slice.token_start_index),
                );
                global_index += 1;
            }

            slice_index += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[SequenceSlice] {
        &self.slices[..self.len]
    }
}

impl Deref for DecodeList {
    type Target = [SequenceSlice];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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

pub struct SequenceState {
    pub sequence_index: usize,
    pub kv_index: usize,
    pub filling_length: usize,
    pub phase: Phase,
    pub notify: Arc<tokio::sync::Notify>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_list_push_preserves_last_token_flag() {
        let mut slices = DecodeList::with_capacity(1);
        slices.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 1,
            last_token_flag: false,
        });

        assert!(!slices[0].last_token_flag);
    }
}
