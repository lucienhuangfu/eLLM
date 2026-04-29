use std::ops::Deref;

#[derive(Clone)]
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
}

impl DecodeList {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slices: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, slice: SequenceSlice) {
        self.slices.push(slice);
    }

    pub fn clear(&mut self) {
        self.slices.clear();
    }

    pub fn total_token_count(&self) -> usize {
        self.slices.iter().map(|slice| slice.length).sum()
    }

    pub fn lookup_global_index(&self, global_index: usize) -> Option<DecodeLookupResult> {
        let slice_index = self
            .slices
            .partition_point(|slice| slice.token_start_index + slice.length <= global_index);
        let slice = self.slices.get(slice_index)?;
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
}

impl Deref for DecodeList {
    type Target = [SequenceSlice];

    fn deref(&self) -> &Self::Target {
        &self.slices
    }
}

#[cfg(test)]
mod tests {
    use super::{DecodeList, DecodeLookupResult, SequenceSlice};

    fn sample_slices() -> DecodeList {
        let mut slices = DecodeList::with_capacity(2);
        slices.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 6,
            last_token_flag: false,
        });
        slices.push(SequenceSlice {
            batch_index: 1,
            sequence_index: 0,
            token_start_index: 6,
            length: 2,
            last_token_flag: false,
        });
        slices
    }

    #[test]
    fn lookup_global_index_returns_decode_lookup_result() {
        let slices = sample_slices();

        assert_eq!(
            slices.lookup_global_index(7),
            Some(DecodeLookupResult {
                slice_index: 1,
                batch_index: 1,
                sequence_index: 1,
            })
        );
        assert_eq!(slices.lookup_global_index(8), None);
    }

    #[test]
    fn walk_global_range_advances_across_slice_boundaries() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        slices.walk_global_range(4, 8, |global_index, batch_index, sequence_index| {
            visited.push((global_index, batch_index, sequence_index));
        });

        assert_eq!(visited, vec![(4, 0, 4), (5, 0, 5), (6, 1, 0), (7, 1, 1)]);
    }

    #[test]
    fn total_token_count_sums_slice_lengths() {
        let mut slices = DecodeList::with_capacity(2);
        slices.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 10,
            length: 6,
            last_token_flag: false,
        });
        slices.push(SequenceSlice {
            batch_index: 1,
            sequence_index: 0,
            token_start_index: 20,
            length: 2,
            last_token_flag: false,
        });

        assert_eq!(slices.total_token_count(), 8);
    }

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
