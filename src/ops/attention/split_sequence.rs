#[inline]
fn triangle_prefix(rows: usize) -> u128 {
    let r = rows as u128;
    r * (r + 1) / 2
}

#[inline]
fn triangle_lower_bound(target: u128, len: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = len;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if triangle_prefix(mid) < target {
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
    thread_num: usize,
    thread_id: usize,
) -> Option<(usize, usize)> {
    if len == 0 || thread_num == 0 || thread_id >= thread_num {
        return None;
    }

    let total = triangle_prefix(len);
    let tn = thread_num as u128;
    let tid = thread_id as u128;
    let work_begin = total * tid / tn;
    let work_end = total * (tid + 1) / tn;

    let begin = triangle_lower_bound(work_begin, len);
    let end = triangle_lower_bound(work_end, len);
    if begin < end {
        Some((begin, end))
    } else {
        None
    }
}
