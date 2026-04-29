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
}
