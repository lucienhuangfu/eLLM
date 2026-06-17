use std::sync::Arc;
use std::time::Instant;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::state::types::SequenceState;

/// 槽位分配器
///
/// 负责管理槽位的 LRU 分配，所有槽位默认在 LRU 池中
pub struct SlotAllocator {
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    slot_timestamps: Arc<SharedMut<Vec<Instant>>>,
}

impl SlotAllocator {
    /// 创建新的槽位分配器
    pub fn new(batch_states: Arc<SharedMut<Vec<SequenceState>>>) -> Self {
        let num_slots = batch_states.with(|states| states.len());
        let timestamps = vec![Instant::now(); num_slots];

        Self {
            batch_states,
            slot_timestamps: Arc::new(SharedMut::new(timestamps)),
        }
    }

    /// 分配最久未使用的槽位
    ///
    /// 时间复杂度: O(n)
    pub fn allocate(&self) -> usize {
        self.slot_timestamps.with(|timestamps| {
            timestamps
                .iter()
                .enumerate()
                .min_by_key(|(_, &ts)| ts)
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
    }

    /// 更新槽位访问时间戳
    pub fn touch(&self, slot_index: usize) {
        self.slot_timestamps.with_mut(|timestamps| {
            if slot_index < timestamps.len() {
                timestamps[slot_index] = Instant::now();
            }
        });
    }

    /// 释放槽位
    pub fn release(&self, slot_index: usize) {
        self.batch_states.with_mut(|batch_list| {
            if let Some(record) = batch_list.get_mut(slot_index) {
                record.reset_to_start();
            }
        });
    }

    /// 获取总槽位数
    pub fn total_slots(&self) -> usize {
        self.batch_states
            .with(|batch_states_ref| batch_states_ref.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_batch_states() -> Arc<SharedMut<Vec<SequenceState>>> {
        Arc::new(SharedMut::new(vec![SequenceState::new_start_state(); 4]))
    }

    #[tokio::test]
    async fn test_allocate_lru() {
        let batch_states = create_test_batch_states();
        let allocator = SlotAllocator::new(batch_states);

        assert_eq!(allocator.total_slots(), 4);

        let slot1 = allocator.allocate();
        assert_eq!(slot1, 0);
        allocator.touch(slot1);

        let slot2 = allocator.allocate();
        assert_eq!(slot2, 1);
        allocator.touch(slot2);

        let slot3 = allocator.allocate();
        assert_eq!(slot3, 2);
        allocator.touch(slot3);

        let slot4 = allocator.allocate();
        assert_eq!(slot4, 3);
        allocator.touch(slot4);

        let slot5 = allocator.allocate();
        assert_eq!(slot5, 0);
    }
}
