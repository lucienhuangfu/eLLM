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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceChannelTileAssign {
    pub slice_index: usize, // Which sequence slice owns this thread.
    pub local_id: usize,    // Thread index inside the slice's thread group.
    pub local_num: usize,   // Total threads assigned to that slice.
}

// Proportional thread distribution over slices, generalizing
// assign_kqv_tile from 3 fixed paths to N slices weighted by row count.
// Every non-empty slice gets at least one thread, then each remaining
// thread goes to the slice with the largest marginal gain
// L/k - L/(k+1) (k = threads already assigned), capped per slice by
// max_blocks (the maximum number of channel blocks, so each block stays
// SIMD friendly). Equal gains fall to the earliest slice, which keeps the
// mapping deterministic for SPMD execution.
// 按行数为 N 个 slice 比例分配线程（assign_kqv_tile 从 3 条固定路径到
// N 个 slice 的推广）：每个非空 slice 至少 1 个线程，之后每个剩余线程
// 都补给边际收益最大的 slice，单 slice 线程数
// 上限为 max_blocks（通道块数上限，保证每块通道数对 SIMD 友好）。收益相等时归给最靠前的 slice，保证 SPMD 执行下分配确定一致。
pub fn assign_slice_channel_tile(
    slice_lengths: &[usize],
    max_blocks: &[usize],
    num: usize,
    id: usize,
) -> Option<SliceChannelTileAssign> {
    debug_assert!(num != 0);
    debug_assert!(id < num);
    debug_assert_eq!(slice_lengths.len(), max_blocks.len());

    let non_empty: Vec<usize> = (0..slice_lengths.len())
        .filter(|&i| slice_lengths[i] > 0)
        .collect();
    if non_empty.is_empty() {
        return None;
    }

    let mut thread_counts = vec![0usize; slice_lengths.len()];
    let mut remaining_threads = num;
    for &i in &non_empty {
        if remaining_threads == 0 {
            break;
        }
        thread_counts[i] = 1;
        remaining_threads -= 1;
    }

    while remaining_threads > 0 {
        // Marginal gain of one more thread on slice i with k threads is
        // L/k - L/(k+1) = L / (k * (k + 1)); compare fractions exactly
        // via cross multiplication to avoid float rounding.
        // slice i 已有 k 个线程时，再加 1 个线程的边际收益为
        // L/k - L/(k+1) = L / (k * (k + 1))；用交叉乘法精确比较分数，
        // 避免浮点舍入。
        let mut best_idx = None;
        let mut best_num = 0u64;
        let mut best_den = 1u64;
        for &i in &non_empty {
            let count = thread_counts[i];
            if count >= max_blocks[i] {
                continue;
            }
            let num = slice_lengths[i] as u64;
            let den = (count * (count + 1)) as u64;
            // num / den > best_num / best_den  <=>  num * best_den > best_num * den
            if best_idx.is_none() || num * best_den > best_num * den {
                best_idx = Some(i);
                best_num = num;
                best_den = den;
            }
        }

        let Some(i) = best_idx else {
            break;
        };
        thread_counts[i] += 1;
        remaining_threads -= 1;
    }

    let mut thread_base = 0usize;
    for i in 0..slice_lengths.len() {
        let count = thread_counts[i];
        if count == 0 {
            continue;
        }
        if id < thread_base + count {
            return Some(SliceChannelTileAssign {
                slice_index: i,
                local_id: id - thread_base,
                local_num: count,
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

    #[test]
    fn test_assign_slice_channel_tile_proportional() {
        // One long prefill slice + two decode slices, 4 threads:
        // the long slice should take 2 threads, the short ones 1 each.
        // 一个长 prefill slice + 两个 decode slice，共 4 线程：
        // 长 slice 应分到 2 个线程，两个短 slice 各 1 个。
        let lengths = [128, 1, 1];
        let max_blocks = [32, 32, 32];
        assert_eq!(
            assign_slice_channel_tile(&lengths, &max_blocks, 4, 0),
            Some(SliceChannelTileAssign {
                slice_index: 0,
                local_id: 0,
                local_num: 2
            })
        );
        assert_eq!(
            assign_slice_channel_tile(&lengths, &max_blocks, 4, 1),
            Some(SliceChannelTileAssign {
                slice_index: 0,
                local_id: 1,
                local_num: 2
            })
        );
        assert_eq!(
            assign_slice_channel_tile(&lengths, &max_blocks, 4, 2),
            Some(SliceChannelTileAssign {
                slice_index: 1,
                local_id: 0,
                local_num: 1
            })
        );
        assert_eq!(
            assign_slice_channel_tile(&lengths, &max_blocks, 4, 3),
            Some(SliceChannelTileAssign {
                slice_index: 2,
                local_id: 0,
                local_num: 1
            })
        );
    }

    #[test]
    fn test_assign_slice_channel_tile_caps_and_overflow() {
        // Two decode slices with cap 1: the 3rd thread idles, no slice left.
        // 两个 decode slice、单 slice 上限 1：第 3 个线程无 slice 可分。
        let lengths = [1, 1];
        let max_blocks = [1, 1];
        assert_eq!(
            assign_slice_channel_tile(&lengths, &max_blocks, 4, 0),
            Some(SliceChannelTileAssign {
                slice_index: 0,
                local_id: 0,
                local_num: 1
            })
        );
        assert_eq!(
            assign_slice_channel_tile(&lengths, &max_blocks, 4, 1),
            Some(SliceChannelTileAssign {
                slice_index: 1,
                local_id: 0,
                local_num: 1
            })
        );
        assert_eq!(assign_slice_channel_tile(&lengths, &max_blocks, 4, 2), None);

        // All slices empty: nothing to schedule.
        // 所有 slice 为空：无可调度任务。
        assert_eq!(assign_slice_channel_tile(&[0, 0], &[32, 32], 2, 0), None);
    }

    #[test]
    fn test_assign_slice_channel_tile_pure_decode() {
        // Pure decode: 4 single-token slices, 8 threads. Equal marginal
        // gains rotate the threads across slices, ending with 2 threads
        // per slice splitting the channel dimension.
        // 纯 decode：4 个单 token slice、8 线程。边际收益相等时线程在各
        // slice 间轮流补给，最终每 slice 2 个线程切分通道。
        let lengths = [1, 1, 1, 1];
        let max_blocks = [4, 4, 4, 4];
        for id in 0..8 {
            let tile = assign_slice_channel_tile(&lengths, &max_blocks, 8, id).unwrap();
            assert_eq!(tile.slice_index, id / 2);
            assert_eq!(tile.local_id, id % 2);
            assert_eq!(tile.local_num, 2);
        }
    }

    #[test]
    fn test_assign_slice_channel_tile_single_long_prefill() {
        // batch=1 long prefill: all threads share the single slice's
        // channels; work ≈ length / thread_num per thread.
        // batch=1 长 prefill：全部线程切分唯一 slice 的通道，
        // 每线程工作量 ≈ length / 线程数。
        let lengths = [256];
        let max_blocks = [8];
        for id in 0..8 {
            assert_eq!(
                assign_slice_channel_tile(&lengths, &max_blocks, 16, id),
                Some(SliceChannelTileAssign {
                    slice_index: 0,
                    local_id: id,
                    local_num: 8
                })
            );
        }
        assert_eq!(
            assign_slice_channel_tile(&lengths, &max_blocks, 16, 8),
            None
        );
    }
}
