use super::task::SequenceSlice;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeLookupResult {
    pub batch_index: usize,
    pub next_sequence_index: usize,
    pub slice_index: usize,
}

pub fn total_sequence_length(slices: &[SequenceSlice]) -> usize {
    slices.iter().map(|s| s.length).sum()
}

pub fn lookup_global_index(
    slices: &[SequenceSlice],
    global_index: usize,
) -> Option<DecodeLookupResult> {
    let slice_index =
        slices.partition_point(|slice| slice.token_start_index + slice.length <= global_index);
    let slice = slices.get(slice_index)?;
    if global_index < slice.token_start_index {
        return None;
    }

    Some(DecodeLookupResult {
        batch_index: slice.batch_index,
        next_sequence_index: slice.next_sequence_index + (global_index - slice.token_start_index),
        slice_index,
    })
}

pub fn walk_global_range(
    slices: &[SequenceSlice],
    global_begin: usize,
    global_end: usize,
    mut visit: impl FnMut(usize, usize, usize),
) {
    if global_begin >= global_end {
        return;
    }

    let Some(found) = lookup_global_index(slices, global_begin) else {
        return;
    };

    let mut slice_index = found.slice_index;
    let mut global_index = global_begin;

    while global_index < global_end {
        let Some(slice) = slices.get(slice_index) else {
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
                slice.next_sequence_index + (global_index - slice.token_start_index),
            );
            global_index += 1;
        }

        slice_index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::task::SequenceSlice;

    fn sample_slices() -> Vec<SequenceSlice> {
        vec![
            SequenceSlice {
                batch_index: 0,
                next_sequence_index: 0,
                token_start_index: 0,
                length: 6,
                last_token_flag: false,
                left_index: 0,
            },
            SequenceSlice {
                batch_index: 1,
                next_sequence_index: 0,
                token_start_index: 6,
                length: 2,
                last_token_flag: false,
                left_index: 0,
            },
        ]
    }

    #[test]
    fn test_push_clear_reuse() {
        let mut list = Vec::with_capacity(2);
        list.push(SequenceSlice {
            batch_index: 0,
            length: 5,
            ..Default::default()
        });
        list.push(SequenceSlice {
            batch_index: 1,
            length: 3,
            ..Default::default()
        });
        assert_eq!(list.len(), 2);
        assert_eq!(total_sequence_length(&list), 8);

        list.clear();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());

        list.push(SequenceSlice {
            batch_index: 2,
            length: 10,
            ..Default::default()
        });
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].batch_index, 2);
    }

    #[test]
    fn test_push_exceeds_capacity() {
        let mut list = Vec::with_capacity(1);
        list.push(SequenceSlice::default());
        list.push(SequenceSlice::default());
        list.push(SequenceSlice::default());
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_lookup_global_index() {
        let slices = sample_slices();

        assert_eq!(
            lookup_global_index(&slices, 0),
            Some(DecodeLookupResult {
                slice_index: 0,
                batch_index: 0,
                next_sequence_index: 0
            })
        );
        assert_eq!(
            lookup_global_index(&slices, 5),
            Some(DecodeLookupResult {
                slice_index: 0,
                batch_index: 0,
                next_sequence_index: 5
            })
        );
        assert_eq!(
            lookup_global_index(&slices, 6),
            Some(DecodeLookupResult {
                slice_index: 1,
                batch_index: 1,
                next_sequence_index: 0
            })
        );
        assert_eq!(
            lookup_global_index(&slices, 7),
            Some(DecodeLookupResult {
                slice_index: 1,
                batch_index: 1,
                next_sequence_index: 1
            })
        );
        assert_eq!(lookup_global_index(&slices, 8), None);
    }

    #[test]
    fn test_lookup_boundary_and_empty() {
        let empty: Vec<SequenceSlice> = Vec::new();
        assert_eq!(lookup_global_index(&empty, 0), None);

        let list = vec![SequenceSlice {
            token_start_index: 10,
            length: 5,
            batch_index: 0,
            next_sequence_index: 100,
            last_token_flag: false,
            left_index: 0,
        }];
        assert_eq!(lookup_global_index(&list, 9), None);
        assert_eq!(
            lookup_global_index(&list, 10),
            Some(DecodeLookupResult {
                slice_index: 0,
                batch_index: 0,
                next_sequence_index: 100
            })
        );
        assert_eq!(lookup_global_index(&list, 15), None);
    }

    #[test]
    fn test_walk_global_range() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        walk_global_range(&slices, 4, 8, |g, b, s| visited.push((g, b, s)));
        assert_eq!(visited, vec![(4, 0, 4), (5, 0, 5), (6, 1, 0), (7, 1, 1)]);
    }

    #[test]
    fn test_walk_empty_and_reverse_range() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        walk_global_range(&slices, 5, 5, |g, b, s| visited.push((g, b, s)));
        assert!(visited.is_empty());

        walk_global_range(&slices, 5, 0, |g, b, s| visited.push((g, b, s)));
        assert!(visited.is_empty());

        walk_global_range(&slices, 100, 200, |g, b, s| visited.push((g, b, s)));
        assert!(visited.is_empty());
    }

    #[test]
    fn test_walk_partial_range() {
        let slices = sample_slices();
        let mut visited = Vec::new();
        walk_global_range(&slices, 0, 100, |g, b, s| visited.push((g, b, s)));
        assert_eq!(visited.len(), 8);
    }
}
