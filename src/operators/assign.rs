// this is a helper function that help locate the partial tasks for the current thread
// parse the range [0, length) evenly into num parts, and return the begin and end of the id-th part
// num is positive, id is 0-indexed, in range [0, num)
// let return tuple be (l, r); elements in range [l, r) belong to the id-th part

// when the length is not a multiple of num, the remain is distributed to the first remain parts
// length
// when length = 10, num = 3, then the parts have 4, 3, 3 elements respectively, the tuples are (0, 4), (4, 7), (7, 10)
// when length = 11, num = 3, then the parts have 4, 4, 3 elements respectively, the tuples are (0, 4), (4, 8), (8, 11)
// when length = 12, num = 3, then the parts have 4, 4, 4 elements respectively, the tuples are (0, 4), (4, 8), (8, 12)
pub fn assign(length: usize, num: usize, id: usize) -> Option<(usize, usize)> {
    debug_assert!(num != 0);
    debug_assert!(id < num);

    if length < (id + 1) {
        return None;
    }

    let (quotient, remainder) = (length / num, length % num);
    // when the length is a multiple of num
    if remainder == 0 {
        let begin = quotient * id;
        let end = begin + quotient;
        return Some((begin, end));
    }

    // when the length is not a multiple of num
    // the remainder is evenlydistributed to the first remainder parts
    if id < remainder {
        let begin = (quotient + 1) * id;
        let end = begin + (quotient + 1);
        return Some((begin, end));
    }

    let begin = quotient * id + remainder;
    let end = begin + quotient;
    return Some((begin, end));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KqvPath {
    V,
    K,
    Q,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KqvTileAssign {
    pub path: KqvPath,
    pub begin: usize,
    pub end: usize,
}

// Minimal K/Q/V merged scheduling:
// each thread belongs to exactly one path, then uses the original assign()
// within that path's local tile space.
pub fn assign_kqv_tile(
    v_length: usize,
    k_length: usize,
    q_length: usize,
    num: usize,
    id: usize,
) -> Option<KqvTileAssign> {
    debug_assert!(num != 0);
    debug_assert!(id < num);

    let total = v_length + k_length + q_length;
    if total == 0 {
        return None;
    }

    let used_threads = num.min(total);
    if id >= used_threads {
        return None;
    }

    let lengths = [v_length, k_length, q_length];
    let paths = [KqvPath::V, KqvPath::K, KqvPath::Q];
    let mut thread_counts = [0usize; 3];
    let mut remaining_threads = used_threads;
    for i in 0..3 {
        if remaining_threads == 0 {
            break;
        }
        if lengths[i] > 0 {
            thread_counts[i] = 1;
            remaining_threads -= 1;
        }
    }

    while remaining_threads > 0 {
        let mut best_idx = None;
        let mut best_gap = 0usize;
        for i in 0..3 {
            let len = lengths[i];
            let count = thread_counts[i];
            if len == 0 || count >= len {
                continue;
            }
            let gap = len.saturating_sub(count);
            if best_idx.is_none() || gap > best_gap {
                best_idx = Some(i);
                best_gap = gap;
            }
        }

        let Some(i) = best_idx else {
            break;
        };
        thread_counts[i] += 1;
        remaining_threads -= 1;
    }
    debug_assert_eq!(thread_counts.iter().sum::<usize>(), used_threads);

    let mut thread_base = 0usize;
    for i in 0..3 {
        let count = thread_counts[i];
        if id < thread_base + count {
            let local_id = id - thread_base;
            let (begin, end) = assign(lengths[i], count, local_id)?;
            return Some(KqvTileAssign {
                path: paths[i],
                begin,
                end,
            });
        }
        thread_base += count;
    }

    None
}

#[cfg(test)]
mod test {
    // use std::result;
    // use approx::assert_ulps_eq;
    use super::*;

    // test assign method
    #[test]
    fn test_assign() {
        let length = 5;
        let num = 8;

        let result = assign(length, num, 5);
        assert_eq!(result, None);
        let result = assign(length, num, 4);
        assert_eq!(result, Some((4, 5)));

        let length = 10;
        let num = 3;

        let result = assign(length, num, 0);
        assert_eq!(result, Some((0, 4)));

        let result = assign(length, num, 0);
        assert_eq!(result, Some((0, 4)));

        let result = assign(length, num, 1);
        assert_eq!(result, Some((4, 7)));

        let result = assign(length, num, 2);
        assert_eq!(result, Some((7, 10)));

        let length = 11;
        let num = 3;
        let result = assign(length, num, 0);
        assert_eq!(result, Some((0, 4)));

        let result = assign(length, num, 1);
        assert_eq!(result, Some((4, 8)));

        let result = assign(length, num, 2);
        assert_eq!(result, Some((8, 11)));

        let length = 12;
        let num = 3;
        let result = assign(length, num, 0);
        assert_eq!(result, Some((0, 4)));

        let result = assign(length, num, 1);
        assert_eq!(result, Some((4, 8)));

        let result = assign(length, num, 2);
        assert_eq!(result, Some((8, 12)));
    }

    #[test]
    fn test_assign_kqv_tile_one_kind_per_thread() {
        assert_eq!(
            assign_kqv_tile(4, 4, 32, 4, 0),
            Some(KqvTileAssign {
                path: KqvPath::V,
                begin: 0,
                end: 4
            })
        );
        assert_eq!(
            assign_kqv_tile(4, 4, 32, 4, 1),
            Some(KqvTileAssign {
                path: KqvPath::K,
                begin: 0,
                end: 4
            })
        );
        assert_eq!(
            assign_kqv_tile(4, 4, 32, 4, 2),
            Some(KqvTileAssign {
                path: KqvPath::Q,
                begin: 0,
                end: 16
            })
        );
        assert_eq!(
            assign_kqv_tile(4, 4, 32, 4, 3),
            Some(KqvTileAssign {
                path: KqvPath::Q,
                begin: 16,
                end: 32
            })
        );
    }

    #[test]
    fn test_assign_kqv_tile_extra_thread_gets_none() {
        assert_eq!(assign_kqv_tile(1, 1, 1, 5, 3), None);
        assert_eq!(assign_kqv_tile(1, 1, 1, 5, 4), None);
    }

    #[test]
    fn test_assign_kqv_tile_skips_empty_segments() {
        assert_eq!(
            assign_kqv_tile(0, 4, 8, 3, 0),
            Some(KqvTileAssign {
                path: KqvPath::K,
                begin: 0,
                end: 4
            })
        );
        assert_eq!(
            assign_kqv_tile(0, 4, 8, 3, 1),
            Some(KqvTileAssign {
                path: KqvPath::Q,
                begin: 0,
                end: 4
            })
        );
        assert_eq!(
            assign_kqv_tile(0, 4, 8, 3, 2),
            Some(KqvTileAssign {
                path: KqvPath::Q,
                begin: 4,
                end: 8
            })
        );
    }
}
