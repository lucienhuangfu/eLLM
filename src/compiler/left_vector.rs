use std::ptr;

use crate::compiler::assign::assign;
use crate::init::record::TokenList;
use crate::init::send_sync_ptr::{ConstPtr, MutPtr};

/// `LiftVector` 用于在推理过程中处理向量数据的搬运（Lifting）。
/// 它主要用于将 Prefill 阶段结束时的状态（向量）复制到新的位置，以便后续 Decode 阶段使用。
/// 这种操作通常发生在 Prefill 完成后，需要将生成的最后一个 token 的状态作为 Decode 的初始状态。
#[derive(Clone)]
pub struct LiftVector<T> {
    /// 指向 TokenList 的常量指针，包含调度所需的记录信息
    token_list_ptr: ConstPtr<TokenList>,
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
    /// * `token_list_ptr` - TokenList 的原始指针。
    /// * `ptr` - 向量数据的原始可变指针。
    /// * `length` - 向量维度。
    pub fn new(token_list_ptr: *const TokenList, ptr: *mut T, length: usize) -> Self {
        Self {
            token_list_ptr: ConstPtr {
                ptr: token_list_ptr,
            },
            ptr: MutPtr { ptr },
            length,
        }
    }

    /// 执行向量搬运操作。
    ///
    /// 该方法根据 `TokenList` 中的 `lift_records` 记录，将数据从 `prefill_end_index` 位置
    /// 复制到 `lift_index` 位置。操作是并行的，根据线程 ID 分配任务。
    ///
    /// # 参数
    /// * `prefill_size` - 当前 token 总数（未使用）。
    /// * `decode_size` - 解码大小（未使用）。
    /// * `thread_num` - 总线程数，用于任务划分。
    /// * `thread_id` - 当前线程 ID。
    pub fn run(&self, prefill_size: usize, decode_size: usize, thread_num: usize, thread_id: usize) {
        let token_list_ptr = self.token_list_ptr.ptr;
        // 获取需要进行 lift 操作的记录数量
        let lift_size = unsafe { (*token_list_ptr).current_lift_size };

        // 根据线程 ID 计算当前线程需要处理的任务范围 [begin, end)
        if let Some((begin, end)) = assign(lift_size, thread_num, thread_id) {
            unsafe {
                let ptr = self.ptr.ptr;
                let records = &(*token_list_ptr).lift_records;

                // 遍历分配给当前线程的记录
                for i in begin..end {
                    let lift_index = records[i].lift_index;
                    let prefill_index = records[i].prefill_end_index;

                    // 计算源地址：prefill 结束位置的向量起始地址
                    let source_ptr = ptr.add(prefill_index * self.length);
                    // 计算目标地址：lift 目标位置的向量起始地址
                    let destination_ptr = ptr.add(lift_index * self.length);

                    // 执行内存复制，将向量从源位置复制到目标位置
                    // copy_nonoverlapping 类似于 C 的 memcpy，假设内存区域不重叠
                    ptr::copy_nonoverlapping(source_ptr, destination_ptr, self.length);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::init::record::{PrefillEndRecord, TokenRecord};

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

        let lift_records = vec![
            PrefillEndRecord {
                prefill_end_index: 2,
                lift_index: 0,
            },
            PrefillEndRecord {
                prefill_end_index: 3,
                lift_index: 1,
            },
        ];

        // 构造 TokenList，token_records 为空即可，因为 run 方法只用 lift_records
        let token_list = TokenList {
            token_records: Box::new([]),
            current_token_size: 0,
            lift_records: lift_records.into_boxed_slice(),
            current_lift_size: 2,
        };

        let lift_vector = LiftVector::new(&token_list, data.as_mut_ptr(), length);

        // 执行搬运，单线程 (thread_num=1, thread_id=0)
        lift_vector.run(0, 0, 1, 0);

        // 验证 Index 0 是否等于 Index 2
        assert_eq!(data[0..4], [1.0, 2.0, 3.0, 4.0]);
        // 验证 Index 1 是否等于 Index 3
        assert_eq!(data[4..8], [5.0, 6.0, 7.0, 8.0]);
    }
}

