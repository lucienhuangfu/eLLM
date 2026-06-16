use std::collections::VecDeque;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::{SlotError, SlotResult};
use crate::runtime::state::types::SequenceState;

/// 简化的槽位分配器 - 只管理空闲槽位队列
pub struct SlotAllocator {
    free_slots: Arc<Mutex<VecDeque<usize>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
}

impl SlotAllocator {
    pub fn new(batch_states: Arc<SharedMut<Vec<SequenceState>>>) -> Self {
        let initial_free_slots: VecDeque<usize> = batch_states.with(|batch_states_ref| {
            batch_states_ref
                .iter()
                .enumerate()
                .filter_map(|(i, record)| record.is_available().then_some(i))
                .collect()
        });

        Self {
            free_slots: Arc::new(Mutex::new(initial_free_slots)),
            batch_states,
        }
    }

    /// 分配一个空闲槽位
    pub async fn allocate(&self) -> SlotResult<usize> {
        let mut free_slots = self.free_slots.lock().await;
        free_slots.pop_front().ok_or(SlotError::SlotQueueEmpty)
    }

    /// 释放槽位，重置状态并放回空闲队列
    pub async fn release(&self, slot_index: usize) {
        // 重置槽位状态
        self.batch_states.with_mut(|batch_list| {
            if let Some(record) = batch_list.get_mut(slot_index) {
                record.reset_to_start();
            }
        });

        // 放回空闲队列
        let mut free_slots = self.free_slots.lock().await;
        free_slots.push_back(slot_index);
    }

    /// 获取可用槽位数量
    pub async fn available_count(&self) -> usize {
        let free_slots = self.free_slots.lock().await;
        free_slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_batch_states() -> Arc<SharedMut<Vec<SequenceState>>> {
        Arc::new(SharedMut::new(vec![SequenceState::new_start_state(); 4]))
    }

    #[tokio::test]
    async fn test_allocate_and_release() {
        let batch_states = create_test_batch_states();
        let allocator = SlotAllocator::new(batch_states);

        // 初始应该有 4 个可用槽位
        assert_eq!(allocator.available_count().await, 4);

        // 分配一个槽位
        let slot1 = allocator.allocate().await.unwrap();
        assert_eq!(allocator.available_count().await, 3);

        // 释放槽位
        allocator.release(slot1).await;
        assert_eq!(allocator.available_count().await, 4);

        // 再次分配应该可以重用
        let slot2 = allocator.allocate().await.unwrap();
        assert_eq!(slot2, slot1);
    }

    #[tokio::test]
    async fn test_allocate_all_slots() {
        let batch_states = create_test_batch_states();
        let allocator = SlotAllocator::new(batch_states);

        // 分配所有槽位
        let slot1 = allocator.allocate().await.unwrap();
        let slot2 = allocator.allocate().await.unwrap();
        let slot3 = allocator.allocate().await.unwrap();
        let slot4 = allocator.allocate().await.unwrap();

        assert_eq!(allocator.available_count().await, 0);

        // 再分配应该失败
        let result = allocator.allocate().await;
        assert!(result.is_err());

        // 释放一个槽位
        allocator.release(slot1).await;
        assert_eq!(allocator.available_count().await, 1);

        // 现在可以再次分配
        let slot5 = allocator.allocate().await.unwrap();
        assert_eq!(slot5, slot1);
    }

    #[tokio::test]
    async fn test_slot_state_reset() {
        let batch_states = create_test_batch_states();
        let allocator = SlotAllocator::new(batch_states.clone());

        // 分配槽位
        let slot = allocator.allocate().await.unwrap();

        // 模拟槽位被使用（修改状态）
        batch_states.with_mut(|batch_list| {
            if let Some(record) = batch_list.get_mut(slot) {
                record.phase = crate::runtime::state::types::Phase::Decode;
                record.sequence_index = 10;
            }
        });

        // 释放槽位
        allocator.release(slot).await;

        // 验证状态已重置
        batch_states.with(|batch_list| {
            let record = &batch_list[slot];
            assert_eq!(record.phase, crate::runtime::state::types::Phase::Start);
            assert_eq!(record.sequence_index, usize::MAX);
        });
    }
}
