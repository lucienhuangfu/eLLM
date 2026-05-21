/// Row visit plan for attention computation
/// Defines main aligned rows and tail unaligned rows to process
#[derive(Copy, Clone)]
pub(super) struct RowVisitPlan {
    pub(super) main: Option<(usize, usize)>,
    pub(super) tail: Option<(usize, usize)>,
}

/// Split a range based on triangle number distribution.
/// Used for load balancing triangular computation across threads.
#[inline]
fn triangle_prefix(rows: usize) -> u128 {
    let r = rows as u128;
    r * (r + 1) / 2
}

#[inline]
fn block_prefix(blocks: usize, len: usize, row_size: usize) -> u128 {
    let covered_rows = (blocks * row_size).min(len);
    triangle_prefix(covered_rows)
}

#[inline]
fn block_lower_bound(target: u128, len: usize, row_size: usize) -> usize {
    let block_count = len.div_ceil(row_size);
    let mut lo = 0usize;
    let mut hi = block_count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if block_prefix(mid, len, row_size) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
pub(super) fn split_sequence_by_triangle(
    len: usize,
    row_size: usize,
    thread_num: usize,
    thread_id: usize,
) -> Option<(usize, usize)> {
    if len == 0 || thread_num == 0 || thread_id >= thread_num {
        return None;
    }

    let row_size = row_size.max(1);
    let total = triangle_prefix(len);
    let tn = thread_num as u128;
    let tid = thread_id as u128;
    let work_begin = total * tid / tn;
    let work_end = total * (tid + 1) / tn;
    let begin_block = block_lower_bound(work_begin, len, row_size);
    let end_block = block_lower_bound(work_end, len, row_size);
    let begin = (begin_block * row_size).min(len);
    let end = (end_block * row_size).min(len);
    (begin < end).then_some((begin, end))
}

#[cfg(test)]
mod test {
    use super::split_sequence_by_triangle;

    #[test]
    fn split_sequence_aligns_to_row_size() {
        assert_eq!(split_sequence_by_triangle(10, 4, 2, 0), Some((0, 8)));
        assert_eq!(split_sequence_by_triangle(10, 4, 2, 1), Some((8, 10)));
    }

    #[test]
    fn split_sequence_defaults_zero_row_size_to_one() {
        assert_eq!(split_sequence_by_triangle(5, 0, 1, 0), Some((0, 5)));
    }

    #[test]
    fn split_sequence_handles_trailing_rows_separately() {
        assert_eq!(split_sequence_by_triangle(10, 4, 3, 0), Some((0, 8)));
        assert_eq!(split_sequence_by_triangle(10, 4, 3, 1), None);
        assert_eq!(split_sequence_by_triangle(10, 4, 3, 2), Some((8, 10)));
    }

    #[test]
    fn split_sequence_handles_exact_row_step_multiple() {
        assert_eq!(split_sequence_by_triangle(12, 4, 3, 0), Some((0, 8)));
        assert_eq!(split_sequence_by_triangle(12, 4, 3, 1), Some((8, 12)));
        assert_eq!(split_sequence_by_triangle(12, 4, 3, 2), None);
    }

    #[test]
    fn split_sequence_covers_aligned_rows_without_overlap() {
        let len = 19;
        let row_size = 4;
        let thread_num = 6;
        let aligned_len = len / row_size * row_size;
        let mut ranges = Vec::new();

        for thread_id in 0..thread_num {
            if let Some(range) =
                split_sequence_by_triangle(aligned_len, row_size, thread_num, thread_id)
            {
                ranges.push((thread_id, range));
            }
        }

        assert!(!ranges.is_empty());
        let first_range = ranges.first().unwrap().1;
        let last_range = ranges.last().unwrap().1;
        assert_eq!(first_range.0, 0);
        assert_eq!(last_range.1, aligned_len);

        let mut cursor = 0;
        for (_, (begin, end)) in ranges {
            assert_eq!(begin, cursor);
            assert!(begin < end);
            assert_eq!(begin % row_size, 0);
            assert_eq!(end % row_size, 0);
            cursor = end;
        }
        assert_eq!(cursor, aligned_len);
    }
}
