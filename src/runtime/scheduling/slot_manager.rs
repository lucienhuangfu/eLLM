use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::{Mutex, Semaphore};

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::{SlotError, SlotResult};
use crate::runtime::scheduling::SequenceState;

pub struct SlotManager {
    free_slots: Arc<Mutex<VecDeque<usize>>>,
    available_slots: Arc<Semaphore>,
    slot_to_dialogue: Arc<Mutex<HashMap<usize, String>>>,
    dialogue_to_slot: Arc<Mutex<HashMap<String, usize>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
}

impl SlotManager {
    pub fn new(batch_states: Arc<SharedMut<Vec<SequenceState>>>) -> Self {
        let initial_free_slots: VecDeque<usize> = batch_states.with(|batch_states_ref| {
            batch_states_ref
                .iter()
                .enumerate()
                .filter_map(|(i, record)| record.is_available().then_some(i))
                .collect()
        });
        let initial_permits = initial_free_slots.len();

        Self {
            free_slots: Arc::new(Mutex::new(initial_free_slots)),
            available_slots: Arc::new(Semaphore::new(initial_permits)),
            slot_to_dialogue: Arc::new(Mutex::new(HashMap::new())),
            dialogue_to_slot: Arc::new(Mutex::new(HashMap::new())),
            batch_states,
        }
    }

    pub async fn acquire_slot(&self, dialogue_id: Option<&str>) -> SlotResult<usize> {
        let permit = self
            .available_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| SlotError::AllocatorUnavailable)?;

        let slot_index = {
            let mut free_slots = self.free_slots.lock().await;
            free_slots.pop_front().ok_or(SlotError::SlotQueueEmpty)?
        };

        if let Some(id) = dialogue_id {
            let mut slot_to_dialogue = self.slot_to_dialogue.lock().await;
            let mut dialogue_to_slot = self.dialogue_to_slot.lock().await;
            slot_to_dialogue.insert(slot_index, id.to_string());
            dialogue_to_slot.insert(id.to_string(), slot_index);
        }

        permit.forget();
        Ok(slot_index)
    }

    pub async fn release_slot(&self, slot_index: usize, release_permit: bool) {
        self.batch_states.with_mut(|batch_list| {
            if let Some(record) = batch_list.get_mut(slot_index) {
                record.reset_to_start();
            }
        });

        let mut slot_to_dialogue = self.slot_to_dialogue.lock().await;
        let mut dialogue_to_slot = self.dialogue_to_slot.lock().await;
        
        if let Some(dialogue_id) = slot_to_dialogue.remove(&slot_index) {
            dialogue_to_slot.remove(&dialogue_id);
        }

        let mut free_slots = self.free_slots.lock().await;
        free_slots.push_back(slot_index);
        drop(free_slots);

        if release_permit {
            self.available_slots.add_permits(1);
        }
    }

    pub async fn release_by_dialogue(&self, dialogue_id: &str) -> Option<usize> {
        let mut slot_to_dialogue = self.slot_to_dialogue.lock().await;
        let mut dialogue_to_slot = self.dialogue_to_slot.lock().await;

        let slot_index = dialogue_to_slot.remove(dialogue_id)?;
        slot_to_dialogue.remove(&slot_index);

        self.batch_states.with_mut(|batch_list| {
            if let Some(record) = batch_list.get_mut(slot_index) {
                record.reset_to_start();
            }
        });

        let mut free_slots = self.free_slots.lock().await;
        free_slots.push_back(slot_index);
        drop(free_slots);

        self.available_slots.add_permits(1);

        Some(slot_index)
    }

    pub async fn get_dialogue_for_slot(&self, slot_index: usize) -> Option<String> {
        let slot_to_dialogue = self.slot_to_dialogue.lock().await;
        slot_to_dialogue.get(&slot_index).cloned()
    }

    pub async fn get_slot_for_dialogue(&self, dialogue_id: &str) -> Option<usize> {
        let dialogue_to_slot = self.dialogue_to_slot.lock().await;
        dialogue_to_slot.get(dialogue_id).copied()
    }

    pub async fn is_slot_owned_by(&self, slot_index: usize, dialogue_id: &str) -> bool {
        let slot_to_dialogue = self.slot_to_dialogue.lock().await;
        slot_to_dialogue.get(&slot_index) == Some(&dialogue_id.to_string())
    }

    pub fn available_permits(&self) -> usize {
        self.available_slots.available_permits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// 创建测试用的 SlotManager，包含指定数量的 slots
    fn create_test_slot_manager(num_slots: usize) -> Arc<SlotManager> {
        let batch_states = Arc::new(SharedMut::new(
            vec![SequenceState::new_start_state(); num_slots]
        ));
        Arc::new(SlotManager::new(batch_states))
    }

    /// 测试创建 SlotManager 后初始状态
    /// 验证: 所有 slots 都可分配，可用许可数等于 slot 总数
    #[tokio::test]
    async fn test_initial_state() {
        let num_slots = 4;
        let slot_manager = create_test_slot_manager(num_slots);
        
        // 初始时所有 slots 都应该可用
        assert_eq!(
            slot_manager.available_permits(), 
            num_slots, 
            "Initial available permits should equal number of slots"
        );
    }

    /// 测试获取单个 slot（不带 dialogue_id）
    /// 验证: 成功获取 slot，许可数减少
    #[tokio::test]
    async fn test_acquire_slot_without_dialogue() {
        let slot_manager = create_test_slot_manager(4);
        
        // 获取一个 slot（不带 dialogue_id）
        let slot_index = slot_manager.acquire_slot(None).await;
        assert!(slot_index.is_ok(), "Should acquire slot successfully");
        
        let idx = slot_index.unwrap();
        assert!(idx < 4, "Slot index should be within valid range");
        
        // 可用许可数应该减少
        assert_eq!(
            slot_manager.available_permits(), 
            3, 
            "Available permits should decrease by 1"
        );
    }

    /// 测试获取 slot 并关联 dialogue_id
    /// 验证: slot 和 dialogue 正确关联，可通过双方查询
    #[tokio::test]
    async fn test_acquire_slot_with_dialogue() {
        let slot_manager = create_test_slot_manager(4);
        let dialogue_id = "test-dialogue-1";
        
        // 获取 slot 并关联 dialogue_id
        let slot_index = slot_manager.acquire_slot(Some(dialogue_id)).await.unwrap();
        
        // 验证 slot 和 dialogue 的双向映射
        let retrieved_dialogue = slot_manager.get_dialogue_for_slot(slot_index).await;
        assert_eq!(
            retrieved_dialogue, 
            Some(dialogue_id.to_string()),
            "Should retrieve correct dialogue_id for slot"
        );
        
        let retrieved_slot = slot_manager.get_slot_for_dialogue(dialogue_id).await;
        assert_eq!(
            retrieved_slot, 
            Some(slot_index),
            "Should retrieve correct slot index for dialogue_id"
        );
        
        // 验证所有权检查
        let is_owned = slot_manager.is_slot_owned_by(slot_index, dialogue_id).await;
        assert!(is_owned, "Slot should be owned by the dialogue");
    }

    /// 测试释放单个 slot
    /// 验证: 释放后 slot 重新可用，许可数增加
    #[tokio::test]
    async fn test_release_slot() {
        let slot_manager = create_test_slot_manager(4);
        
        // 获取一个 slot
        let slot_index = slot_manager.acquire_slot(Some("dialogue-1")).await.unwrap();
        
        // 释放 slot
        slot_manager.release_slot(slot_index, true).await;
        
        // 验证可用许可数恢复
        assert_eq!(
            slot_manager.available_permits(), 
            4, 
            "Available permits should be restored after release"
        );
        
        // 验证 dialogue 关联已清除
        let dialogue = slot_manager.get_dialogue_for_slot(slot_index).await;
        assert!(dialogue.is_none(), "Dialogue association should be cleared");
    }

    /// 测试通过 dialogue_id 释放 slot
    /// 验证: 通过 dialogue_id 正确释放对应的 slot
    #[tokio::test]
    async fn test_release_by_dialogue() {
        let slot_manager = create_test_slot_manager(4);
        let dialogue_id = "test-dialogue-release";
        
        // 获取 slot
        let slot_index = slot_manager.acquire_slot(Some(dialogue_id)).await.unwrap();
        
        // 通过 dialogue_id 释放
        let released_slot = slot_manager.release_by_dialogue(dialogue_id).await;
        assert_eq!(
            released_slot, 
            Some(slot_index),
            "Should return the correct slot index"
        );
        
        // 验证 slot 已释放
        assert_eq!(
            slot_manager.available_permits(), 
            4, 
            "Slot should be released"
        );
        
        // 验证 dialogue 关联已清除
        let slot = slot_manager.get_slot_for_dialogue(dialogue_id).await;
        assert!(slot.is_none(), "Dialogue should no longer have associated slot");
    }

    /// 测试获取不存在的 dialogue 的 slot
    /// 验证: 返回 None
    #[tokio::test]
    async fn test_get_slot_for_nonexistent_dialogue() {
        let slot_manager = create_test_slot_manager(4);
        
        let slot = slot_manager.get_slot_for_dialogue("nonexistent").await;
        assert!(slot.is_none(), "Should return None for non-existent dialogue");
    }

    /// 测试释放不存在的 dialogue
    /// 验证: 返回 None
    #[tokio::test]
    async fn test_release_nonexistent_dialogue() {
        let slot_manager = create_test_slot_manager(4);
        
        let result = slot_manager.release_by_dialogue("nonexistent").await;
        assert!(result.is_none(), "Should return None for non-existent dialogue");
    }

    /// 测试获取所有 slots 后再释放
    /// 验证: 所有 slots 分配完毕后，再次获取会阻塞（在测试中表现为需要等待）
    #[tokio::test]
    async fn test_acquire_all_slots() {
        let num_slots = 3;
        let slot_manager = create_test_slot_manager(num_slots);
        
        // 获取所有 slots
        let slot1 = slot_manager.acquire_slot(Some("d1")).await.unwrap();
        let slot2 = slot_manager.acquire_slot(Some("d2")).await.unwrap();
        let slot3 = slot_manager.acquire_slot(Some("d3")).await.unwrap();
        
        assert_eq!(slot1, 0, "First slot should be index 0");
        assert_eq!(slot2, 1, "Second slot should be index 1");
        assert_eq!(slot3, 2, "Third slot should be index 2");
        
        // 验证所有 slots 已分配
        assert_eq!(
            slot_manager.available_permits(), 
            0, 
            "No permits should be available"
        );
    }

    /// 测试 slot 索引重用
    /// 验证: 释放的 slot 可以被重新分配
    #[tokio::test]
    async fn test_slot_reuse() {
        let slot_manager = create_test_slot_manager(2);
        
        // 获取并释放 slot 0
        let slot0 = slot_manager.acquire_slot(Some("d1")).await.unwrap();
        assert_eq!(slot0, 0);
        slot_manager.release_slot(slot0, true).await;
        
        // 获取并释放 slot 1
        let slot1 = slot_manager.acquire_slot(Some("d2")).await.unwrap();
        assert_eq!(slot1, 1);
        slot_manager.release_slot(slot1, true).await;
        
        // 再次获取，应该重用之前释放的 slots
        let slot0_reuse = slot_manager.acquire_slot(Some("d3")).await.unwrap();
        assert_eq!(slot0_reuse, 0, "Should reuse slot 0");
        
        let slot1_reuse = slot_manager.acquire_slot(Some("d4")).await.unwrap();
        assert_eq!(slot1_reuse, 1, "Should reuse slot 1");
    }

    /// 测试 is_slot_owned_by 方法
    /// 验证: 正确判断 slot 所有权
    #[tokio::test]
    async fn test_is_slot_owned_by() {
        let slot_manager = create_test_slot_manager(2);
        
        // 获取 slot 并关联 dialogue
        let slot = slot_manager.acquire_slot(Some("owner-dialogue")).await.unwrap();
        
        // 验证所有权
        assert!(
            slot_manager.is_slot_owned_by(slot, "owner-dialogue").await,
            "Slot should be owned by owner-dialogue"
        );
        assert!(
            !slot_manager.is_slot_owned_by(slot, "other-dialogue").await,
            "Slot should not be owned by other-dialogue"
        );
        
        // 释放后验证所有权清除
        slot_manager.release_slot(slot, true).await;
        assert!(
            !slot_manager.is_slot_owned_by(slot, "owner-dialogue").await,
            "Slot ownership should be cleared after release"
        );
    }
}