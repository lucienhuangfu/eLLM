use std::ops::Deref;

#[derive(Clone, Default, Debug)]
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

#[derive(Clone, Default, Debug)]
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

    /// 测试 SequenceSlice 默认值
    #[test]
    fn test_sequence_slice_default() {
        let slice = SequenceSlice::default();
        assert_eq!(slice.token_start_index, 0);
        assert_eq!(slice.batch_index, 0);
        assert_eq!(slice.sequence_index, 0);
        assert_eq!(slice.length, 0);
        assert!(!slice.last_token_flag);
    }

    /// 测试 SequenceSlice Clone 特性
    #[test]
    fn test_sequence_slice_clone() {
        let slice = SequenceSlice {
            token_start_index: 10,
            batch_index: 2,
            sequence_index: 5,
            length: 3,
            last_token_flag: true,
        };
        let cloned = slice.clone();
        assert_eq!(slice.token_start_index, cloned.token_start_index);
        assert_eq!(slice.batch_index, cloned.batch_index);
        assert_eq!(slice.sequence_index, cloned.sequence_index);
        assert_eq!(slice.length, cloned.length);
        assert_eq!(slice.last_token_flag, cloned.last_token_flag);
    }

    /// 测试 SequenceSlice Debug 实现
    #[test]
    fn test_sequence_slice_debug() {
        let slice = SequenceSlice {
            token_start_index: 0,
            batch_index: 1,
            sequence_index: 2,
            length: 3,
            last_token_flag: true,
        };
        let debug_str = format!("{:?}", slice);
        assert!(debug_str.contains("SequenceSlice"));
        assert!(debug_str.contains("batch_index"));
        assert!(debug_str.contains("length"));
    }

    /// 测试 DecodeLookupResult 创建
    #[test]
    fn test_decode_lookup_result_creation() {
        let result = DecodeLookupResult {
            batch_index: 1,
            sequence_index: 10,
            slice_index: 2,
        };
        assert_eq!(result.batch_index, 1);
        assert_eq!(result.sequence_index, 10);
        assert_eq!(result.slice_index, 2);
    }

    /// 测试 DecodeLookupResult Copy 特性
    #[test]
    fn test_decode_lookup_result_copy() {
        let result = DecodeLookupResult {
            batch_index: 1,
            sequence_index: 10,
            slice_index: 2,
        };
        let copied = result;
        assert_eq!(result.batch_index, copied.batch_index);
        assert_eq!(result.sequence_index, copied.sequence_index);
        assert_eq!(result.slice_index, copied.slice_index);
    }

    /// 测试 DecodeLookupResult PartialEq
    #[test]
    fn test_decode_lookup_result_partial_eq() {
        let result1 = DecodeLookupResult {
            batch_index: 1,
            sequence_index: 10,
            slice_index: 2,
        };
        let result2 = DecodeLookupResult {
            batch_index: 1,
            sequence_index: 10,
            slice_index: 2,
        };
        let result3 = DecodeLookupResult {
            batch_index: 2,
            sequence_index: 10,
            slice_index: 2,
        };
        assert_eq!(result1, result2);
        assert_ne!(result1, result3);
    }

    /// 测试 DecodeList 默认值
    #[test]
    fn test_decode_list_default() {
        let list = DecodeList::default();
        assert_eq!(list.len(), 0);
        assert!(list.as_slice().is_empty());
    }

    /// 测试 DecodeList::with_capacity
    #[test]
    fn test_decode_list_with_capacity() {
        let list = DecodeList::with_capacity(10);
        assert_eq!(list.len(), 0);
        // 内部 slices 容量应该至少为 10
    }

    /// 测试 DecodeList::with_capacity 零容量
    #[test]
    fn test_decode_list_with_capacity_zero() {
        let list = DecodeList::with_capacity(0);
        assert_eq!(list.len(), 0);
    }

    /// 测试 DecodeList::push
    #[test]
    fn test_decode_list_push() {
        let mut list = DecodeList::with_capacity(2);
        list.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 5,
            last_token_flag: false,
        });
        assert_eq!(list.len(), 1);

        list.push(SequenceSlice {
            batch_index: 1,
            sequence_index: 0,
            token_start_index: 5,
            length: 3,
            last_token_flag: true,
        });
        assert_eq!(list.len(), 2);
    }

    /// 测试 DecodeList::push 超出初始容量
    #[test]
    fn test_decode_list_push_exceeds_capacity() {
        let mut list = DecodeList::with_capacity(1);
        list.push(SequenceSlice::default());
        list.push(SequenceSlice::default());
        list.push(SequenceSlice::default());
        assert_eq!(list.len(), 3);
    }

    /// 测试 DecodeList::clear
    #[test]
    fn test_decode_list_clear() {
        let mut list = DecodeList::with_capacity(2);
        list.push(SequenceSlice::default());
        list.push(SequenceSlice::default());
        assert_eq!(list.len(), 2);

        list.clear();
        assert_eq!(list.len(), 0);
        assert!(list.as_slice().is_empty());
    }

    /// 测试 DecodeList::clear 后重新 push
    #[test]
    fn test_decode_list_clear_and_push_again() {
        let mut list = DecodeList::with_capacity(2);
        list.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 5,
            last_token_flag: false,
        });
        list.clear();

        list.push(SequenceSlice {
            batch_index: 1,
            sequence_index: 10,
            token_start_index: 0,
            length: 3,
            last_token_flag: true,
        });
        assert_eq!(list.len(), 1);
        assert_eq!(list.as_slice()[0].batch_index, 1);
    }

    /// 测试 DecodeList::total_token_count
    #[test]
    fn test_decode_list_total_token_count() {
        let mut list = DecodeList::with_capacity(3);
        list.push(SequenceSlice {
            length: 5,
            ..Default::default()
        });
        list.push(SequenceSlice {
            length: 3,
            ..Default::default()
        });
        list.push(SequenceSlice {
            length: 2,
            ..Default::default()
        });
        assert_eq!(list.total_token_count(), 10);
    }

    /// 测试 DecodeList::total_token_count 空列表
    #[test]
    fn test_decode_list_total_token_count_empty() {
        let list = DecodeList::with_capacity(5);
        assert_eq!(list.total_token_count(), 0);
    }

    /// 测试 DecodeList::len
    #[test]
    fn test_decode_list_len() {
        let mut list = DecodeList::with_capacity(5);
        assert_eq!(list.len(), 0);

        for i in 0..5 {
            list.push(SequenceSlice::default());
            assert_eq!(list.len(), i + 1);
        }
    }

    /// 测试 DecodeList::as_slice
    #[test]
    fn test_decode_list_as_slice() {
        let mut list = DecodeList::with_capacity(2);
        list.push(SequenceSlice {
            batch_index: 0,
            ..Default::default()
        });
        list.push(SequenceSlice {
            batch_index: 1,
            ..Default::default()
        });

        let slice = list.as_slice();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0].batch_index, 0);
        assert_eq!(slice[1].batch_index, 1);
    }

    /// 测试 DecodeList Deref 实现
    #[test]
    fn test_decode_list_deref() {
        let mut list = DecodeList::with_capacity(2);
        list.push(SequenceSlice {
            batch_index: 0,
            ..Default::default()
        });
        list.push(SequenceSlice {
            batch_index: 1,
            ..Default::default()
        });

        // 通过 Deref 访问
        assert_eq!(list[0].batch_index, 0);
        assert_eq!(list[1].batch_index, 1);
        assert_eq!(list.len(), 2); // 注意：这是 slice 的 len，不是 DecodeList 的 len
    }

    /// 测试 DecodeList::lookup_global_index 第一个切片
    #[test]
    fn test_decode_list_lookup_first_slice() {
        let slices = sample_slices();
        assert_eq!(
            slices.lookup_global_index(0),
            Some(DecodeLookupResult {
                slice_index: 0,
                batch_index: 0,
                sequence_index: 0,
            })
        );
        assert_eq!(
            slices.lookup_global_index(5),
            Some(DecodeLookupResult {
                slice_index: 0,
                batch_index: 0,
                sequence_index: 5,
            })
        );
    }

    /// 测试 DecodeList::lookup_global_index 第二个切片
    #[test]
    fn test_decode_list_lookup_second_slice() {
        let slices = sample_slices();
        assert_eq!(
            slices.lookup_global_index(6),
            Some(DecodeLookupResult {
                slice_index: 1,
                batch_index: 1,
                sequence_index: 0,
            })
        );
        assert_eq!(
            slices.lookup_global_index(7),
            Some(DecodeLookupResult {
                slice_index: 1,
                batch_index: 1,
                sequence_index: 1,
            })
        );
    }

    /// 测试 DecodeList::lookup_global_index 超出范围
    #[test]
    fn test_decode_list_lookup_out_of_range() {
        let slices = sample_slices();
        assert_eq!(slices.lookup_global_index(8), None);
        assert_eq!(slices.lookup_global_index(100), None);
    }

    /// 测试 DecodeList::lookup_global_index 空列表
    #[test]
    fn test_decode_list_lookup_empty() {
        let list = DecodeList::with_capacity(5);
        assert_eq!(list.lookup_global_index(0), None);
    }

    /// 测试 DecodeList::lookup_global_index 边界情况
    #[test]
    fn test_decode_list_lookup_boundary() {
        let mut list = DecodeList::with_capacity(1);
        list.push(SequenceSlice {
            token_start_index: 10,
            length: 5,
            batch_index: 0,
            sequence_index: 100,
            last_token_flag: false,
        });

        // 刚好在边界上
        assert_eq!(
            list.lookup_global_index(10),
            Some(DecodeLookupResult {
                slice_index: 0,
                batch_index: 0,
                sequence_index: 100,
            })
        );
        assert_eq!(
            list.lookup_global_index(14),
            Some(DecodeLookupResult {
                slice_index: 0,
                batch_index: 0,
                sequence_index: 104,
            })
        );
        // 刚好在边界外
        assert_eq!(list.lookup_global_index(15), None);
        // 在 token_start_index 之前
        assert_eq!(list.lookup_global_index(9), None);
    }

    /// 测试 DecodeList::walk_global_range 单切片
    #[test]
    fn test_decode_list_walk_single_slice() {
        let mut list = DecodeList::with_capacity(1);
        list.push(SequenceSlice {
            token_start_index: 0,
            length: 5,
            batch_index: 0,
            sequence_index: 0,
            last_token_flag: false,
        });

        let mut visited = Vec::new();
        list.walk_global_range(0, 5, |global, batch, seq| {
            visited.push((global, batch, seq));
        });

        assert_eq!(
            visited,
            vec![(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)]
        );
    }

    /// 测试 DecodeList::walk_global_range 跨切片
    #[test]
    fn test_decode_list_walk_cross_slice() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        slices.walk_global_range(4, 8, |global_index, batch_index, sequence_index| {
            visited.push((global_index, batch_index, sequence_index));
        });

        assert_eq!(visited, vec![(4, 0, 4), (5, 0, 5), (6, 1, 0), (7, 1, 1)]);
    }

    /// 测试 DecodeList::walk_global_range 空范围
    #[test]
    fn test_decode_list_walk_empty_range() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        slices.walk_global_range(5, 5, |global, batch, seq| {
            visited.push((global, batch, seq));
        });

        assert!(visited.is_empty());
    }

    /// 测试 DecodeList::walk_global_range 超出范围
    #[test]
    fn test_decode_list_walk_out_of_range() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        slices.walk_global_range(100, 200, |global, batch, seq| {
            visited.push((global, batch, seq));
        });

        assert!(visited.is_empty());
    }

    /// 测试 DecodeList::walk_global_range 部分在范围内
    #[test]
    fn test_decode_list_walk_partial_range() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        // 开始在范围内，结束超出范围
        slices.walk_global_range(0, 100, |global, batch, seq| {
            visited.push((global, batch, seq));
        });

        assert_eq!(visited.len(), 8); // 只有 8 个有效 token
    }

    /// 测试 DecodeList::walk_global_range 反向范围
    #[test]
    fn test_decode_list_walk_reverse_range() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        // end < begin 应该不访问任何元素
        slices.walk_global_range(5, 0, |global, batch, seq| {
            visited.push((global, batch, seq));
        });

        assert!(visited.is_empty());
    }

    /// 测试 DecodeList Clone 特性
    #[test]
    fn test_decode_list_clone() {
        let mut list = DecodeList::with_capacity(2);
        list.push(SequenceSlice {
            batch_index: 0,
            sequence_index: 0,
            token_start_index: 0,
            length: 5,
            last_token_flag: false,
        });

        let cloned = list.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.as_slice()[0].batch_index, 0);
    }

    /// 测试 DecodeList Debug 实现
    #[test]
    fn test_decode_list_debug() {
        let list = DecodeList::with_capacity(2);
        let debug_str = format!("{:?}", list);
        assert!(debug_str.contains("DecodeList"));
    }

    /// 测试 DecodeList 多切片 lookup
    #[test]
    fn test_decode_list_multi_slice_lookup() {
        let mut list = DecodeList::with_capacity(3);
        list.push(SequenceSlice {
            token_start_index: 0,
            length: 3,
            batch_index: 0,
            sequence_index: 0,
            last_token_flag: false,
        });
        list.push(SequenceSlice {
            token_start_index: 3,
            length: 4,
            batch_index: 1,
            sequence_index: 10,
            last_token_flag: false,
        });
        list.push(SequenceSlice {
            token_start_index: 7,
            length: 2,
            batch_index: 2,
            sequence_index: 20,
            last_token_flag: false,
        });

        // 测试每个切片的 lookup
        assert_eq!(
            list.lookup_global_index(0),
            Some(DecodeLookupResult {
                slice_index: 0,
                batch_index: 0,
                sequence_index: 0,
            })
        );
        assert_eq!(
            list.lookup_global_index(5),
            Some(DecodeLookupResult {
                slice_index: 1,
                batch_index: 1,
                sequence_index: 12,
            })
        );
        assert_eq!(
            list.lookup_global_index(8),
            Some(DecodeLookupResult {
                slice_index: 2,
                batch_index: 2,
                sequence_index: 21,
            })
        );
    }

    /// 测试 DecodeList 大量切片
    #[test]
    fn test_decode_list_many_slices() {
        let mut list = DecodeList::with_capacity(0);
        for i in 0..100 {
            list.push(SequenceSlice {
                token_start_index: i * 10,
                length: 10,
                batch_index: i,
                sequence_index: i * 100,
                last_token_flag: i == 99,
            });
        }
        assert_eq!(list.len(), 100);
        assert_eq!(list.total_token_count(), 1000);

        // 测试中间位置的 lookup
        assert_eq!(
            list.lookup_global_index(500),
            Some(DecodeLookupResult {
                slice_index: 50,
                batch_index: 50,
                sequence_index: 5000,
            })
        );
    }

    /// 测试 SequenceSlice last_token_flag 功能
    #[test]
    fn test_sequence_slice_last_token_flag() {
        let mut list = DecodeList::with_capacity(3);
        list.push(SequenceSlice {
            last_token_flag: false,
            ..Default::default()
        });
        list.push(SequenceSlice {
            last_token_flag: true,
            ..Default::default()
        });
        list.push(SequenceSlice {
            last_token_flag: false,
            ..Default::default()
        });

        assert!(!list[0].last_token_flag);
        assert!(list[1].last_token_flag);
        assert!(!list[2].last_token_flag);
    }

    /// 测试 DecodeLookupResult Debug 实现
    #[test]
    fn test_decode_lookup_result_debug() {
        let result = DecodeLookupResult {
            batch_index: 1,
            sequence_index: 10,
            slice_index: 2,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("DecodeLookupResult"));
    }

    /// 测试 DecodeList::walk_global_range 完整遍历
    #[test]
    fn test_decode_list_walk_complete() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        slices.walk_global_range(0, 8, |global, batch, seq| {
            visited.push((global, batch, seq));
        });

        assert_eq!(
            visited,
            vec![
                (0, 0, 0),
                (1, 0, 1),
                (2, 0, 2),
                (3, 0, 3),
                (4, 0, 4),
                (5, 0, 5),
                (6, 1, 0),
                (7, 1, 1)
            ]
        );
    }

    /// 测试 DecodeList 空列表 walk
    #[test]
    fn test_decode_list_walk_empty_list() {
        let list = DecodeList::with_capacity(5);
        let mut visited = Vec::new();

        list.walk_global_range(0, 10, |global, batch, seq| {
            visited.push((global, batch, seq));
        });

        assert!(visited.is_empty());
    }

    /// 测试 DecodeList::lookup_global_index 返回值正确性
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

    /// 测试 DecodeList::walk_global_range 跨切片边界
    #[test]
    fn walk_global_range_advances_across_slice_boundaries() {
        let slices = sample_slices();
        let mut visited = Vec::new();

        slices.walk_global_range(4, 8, |global_index, batch_index, sequence_index| {
            visited.push((global_index, batch_index, sequence_index));
        });

        assert_eq!(visited, vec![(4, 0, 4), (5, 0, 5), (6, 1, 0), (7, 1, 1)]);
    }

    /// 测试 DecodeList::total_token_count 求和
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

    /// 测试 DecodeList::push 保留 last_token_flag
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
