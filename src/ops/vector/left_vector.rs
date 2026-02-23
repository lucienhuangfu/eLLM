use std::ptr;

use crate::serving::record::SequenceSlice;
use crate::common::send_sync_ptr::MutPtr;

/// `LiftVector` 用于在推理过程中处理向量数据的搬运（Lifting）。
/// 它主要用于将 Prefill 阶段结束时的状态（向量）复制到新的位置，以便后续 Decode 阶段使用。
/// 这种操作通常发生在 Prefill 完成后，需要将生成的最后一个 token 的状态作为 Decode 的初始状态。
#[derive(Clone)]
pub struct LiftVector<T> {
    /// 指向实际向量数据的可变指针，数据将在该内存区域内进行复制
    ptr: MutPtr<T>,
    /// 单个向量的长度（维度大小）
    length: usize,
}

// 移除 Sqrt 约束，改为泛型 T
impl<T> LiftVector<T> {
    /// 创建一个新的 `LiftVector` 实例。
    ///
    /// # 参数
    /// * `ptr` - 向量数据的原始可变指针。
    /// * `length` - 向量维度。
    pub fn new(ptr: *mut T, length: usize) -> Self {
        Self {
            ptr: MutPtr { ptr },
            length,
        }
    }

    /// 执行向量搬运操作。
    ///
    /// 该方法根据 `decode_tokens` 中的切片记录，将数据从 `token_start_index` 位置
    /// 复制到 `lift_index` 位置。`decode_tokens` 已经完成任务划分。
    ///
    /// # 参数
    pub fn run(
        &self,
        _prefill_size: usize,
        _decode_size: usize,
        decode_tokens: &[SequenceSlice],
        _thread_num: usize,
        _thread_id: usize,
    ) {
        if _prefill_size > 0 {
            unsafe {
                let ptr = self.ptr.ptr;

                for slice in decode_tokens {
                    for offset in 0..slice.length {
                        let source_index = slice.token_start_index + offset;
                        let destination_index = slice.lift_index + offset;

                        let source_ptr = ptr.add(source_index * self.length);
                        let destination_ptr = ptr.add(destination_index * self.length);

                        // copy_nonoverlapping 类似于 C 的 memcpy，假设内存区域不重叠
                        ptr::copy_nonoverlapping(source_ptr, destination_ptr, self.length);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::serving::record::SequenceSlice;

    #[test]
    fn test_lift_vector() {
        let length = 4;
        // 模拟数据: 2个源向量，2个目标向量位置
        // 修改: prefill_end_index 位于数据的下半部分 (Index 2, 3)
        // Index 0: Dest 1 (should copy from Index 2)
        // Index 1: Dest 2 (should copy from Index 3)
        // Index 2: Source 1
        // Index 3: Source 2
        let mut data: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, // Index 0
            0.0, 0.0, 0.0, 0.0, // Index 1
            1.0, 2.0, 3.0, 4.0, // Index 2
            5.0, 6.0, 7.0, 8.0, // Index 3
        ];

        let decode_tokens = vec![
            SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 2,
                lift_index: 0,
                length: 1,
            },
            SequenceSlice {
                batch_index: 0,
                sequence_index: 0,
                token_start_index: 3,
                lift_index: 1,
                length: 1,
            },
        ];

        let lift_vector = LiftVector::new(data.as_mut_ptr(), length);

        // 执行搬运，任务已分配
        lift_vector.run(0, 0, &decode_tokens, 1, 0);

        // 验证 Index 0 是否等于 Index 2
        assert_eq!(data[0..4], [1.0, 2.0, 3.0, 4.0]);
        // 验证 Index 1 是否等于 Index 3
        assert_eq!(data[4..8], [5.0, 6.0, 7.0, 8.0]);
    }
}
