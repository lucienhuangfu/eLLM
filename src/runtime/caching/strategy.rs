use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, RwLock};

use crate::runtime::scheduling::slot_manager::SlotManager;

use super::lru_list::LruList;

#[derive(Debug, Clone)]
pub struct DialogueEntry {
    pub dialogue_id: String,
    pub slot_index: usize,
    pub token_count: usize,
    pub last_accessed_at: Instant,
    pub in_lru: bool,
    pub lru_index: Option<usize>,
}

pub struct LruCacheStrategy {
    entries: RwLock<std::collections::HashMap<String, DialogueEntry>>,
    lru_list: Mutex<LruList>,
    retention_duration: Duration,
    max_entries: usize,
    slot_manager: Arc<SlotManager>,
}

impl LruCacheStrategy {
    pub fn new(
        slot_manager: Arc<SlotManager>,
        retention_duration: Duration,
        max_entries: usize,
    ) -> Self {
        Self {
            entries: RwLock::new(std::collections::HashMap::new()),
            lru_list: Mutex::new(LruList::new()),
            retention_duration,
            max_entries,
            slot_manager,
        }
    }

    pub async fn get(&self, dialogue_id: &str) -> Option<DialogueEntry> {
        let entries_read = self.entries.read().await;
        let entry = entries_read.get(dialogue_id)?;
        let entry_clone = entry.clone();
        drop(entries_read);

        let mut entries_write = self.entries.write().await;
        let entry = entries_write.get_mut(dialogue_id)?;

        if entry.in_lru {
            let mut lru_list = self.lru_list.lock().await;
            if let Some(lru_index) = entry.lru_index {
                lru_list.remove(lru_index);
            }
            entry.lru_index = None;
            entry.in_lru = false;
        }

        entry.last_accessed_at = Instant::now();

        Some(entry_clone)
    }

    pub async fn insert(&self, dialogue_id: String, slot_index: usize, token_count: usize) {
        let mut entries = self.entries.write().await;

        let entry = DialogueEntry {
            dialogue_id: dialogue_id.clone(),
            slot_index,
            token_count,
            last_accessed_at: Instant::now(),
            in_lru: false,
            lru_index: None,
        };

        entries.insert(dialogue_id, entry);

        drop(entries);

        self.cleanup(Instant::now()).await;
    }

    pub async fn remove(&self, dialogue_id: &str) {
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.remove(dialogue_id) {
            if entry.in_lru {
                let mut lru_list = self.lru_list.lock().await;
                if let Some(lru_index) = entry.lru_index {
                    lru_list.remove(lru_index);
                }
            }

            drop(entries);

            self.slot_manager.release_by_dialogue(dialogue_id).await;
        }
    }

    pub async fn cleanup(&self, now: Instant) {
        let to_evict: Vec<String> = {
            let mut entries = self.entries.write().await;
            let mut lru_list = self.lru_list.lock().await;

            for entry in entries.values_mut() {
                if !entry.in_lru && (now - entry.last_accessed_at) >= self.retention_duration {
                    entry.in_lru = true;
                    let lru_index = lru_list.push_back(entry.dialogue_id.clone());
                    entry.lru_index = Some(lru_index);
                }
            }

            let mut to_evict = Vec::new();
            while entries.len() > self.max_entries && !lru_list.is_empty() {
                if let Some(dialogue_id) = lru_list.pop_back() {
                    if let Some(entry) = entries.remove(&dialogue_id) {
                        to_evict.push(entry.dialogue_id);
                    }
                }
            }

            to_evict
        };

        for dialogue_id in to_evict {
            self.slot_manager.release_by_dialogue(&dialogue_id).await;
        }
    }

    pub async fn entry_count(&self) -> usize {
        self.entries.read().await.len()
    }

    pub async fn insert_batch(&self, entries: Vec<(String, usize, usize)>) {
        let mut lock = self.entries.write().await;
        for (dialogue_id, slot_index, token_count) in entries {
            lock.insert(
                dialogue_id.clone(),
                DialogueEntry {
                    dialogue_id,
                    slot_index,
                    token_count,
                    last_accessed_at: Instant::now(),
                    in_lru: false,
                    lru_index: None,
                },
            );
        }
        drop(lock);
        self.cleanup(Instant::now()).await;
    }

    pub async fn remove_batch(&self, dialogue_ids: &[&str]) {
        let mut to_release: Vec<String> = Vec::new();

        {
            let mut entries = self.entries.write().await;
            let mut lru_list = self.lru_list.lock().await;

            for &dialogue_id in dialogue_ids {
                if let Some(entry) = entries.remove(dialogue_id) {
                    if entry.in_lru {
                        if let Some(lru_index) = entry.lru_index {
                            lru_list.remove(lru_index);
                        }
                    }
                    to_release.push(entry.dialogue_id);
                }
            }
        }

        for dialogue_id in to_release {
            self.slot_manager.release_by_dialogue(&dialogue_id).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::send_sync_ptr::SharedMut;
    use crate::runtime::scheduling::SequenceState;
    use std::sync::Arc;
    use std::time::Duration;

    /// 创建测试用的 SlotManager
    fn create_test_slot_manager(num_slots: usize) -> Arc<SlotManager> {
        let batch_states = Arc::new(SharedMut::new(
            vec![SequenceState::new_start_state(); num_slots]
        ));
        Arc::new(SlotManager::new(batch_states))
    }

    /// 测试插入和获取缓存条目
    /// 验证: 插入条目后可以正确获取，且条目信息完整
    #[tokio::test]
    async fn test_insert_and_get() {
        let slot_manager = create_test_slot_manager(4);
        let strategy = LruCacheStrategy::new(
            slot_manager,
            Duration::from_secs(60),
            10
        );

        // 插入条目
        strategy.insert("dialogue-1".to_string(), 0, 100).await;

        // 获取条目
        let entry = strategy.get("dialogue-1").await;
        assert!(entry.is_some(), "Should retrieve inserted entry");
        
        let entry = entry.unwrap();
        assert_eq!(entry.dialogue_id, "dialogue-1");
        assert_eq!(entry.slot_index, 0);
        assert_eq!(entry.token_count, 100);
        assert!(!entry.in_lru);
        assert!(entry.lru_index.is_none());
    }

    /// 测试获取不存在的条目
    /// 验证: 返回 None
    #[tokio::test]
    async fn test_get_nonexistent() {
        let slot_manager = create_test_slot_manager(4);
        let strategy = LruCacheStrategy::new(
            slot_manager,
            Duration::from_secs(60),
            10
        );

        let entry = strategy.get("nonexistent").await;
        assert!(entry.is_none(), "Should return None for non-existent dialogue");
    }

    /// 测试删除条目
    /// 验证: 删除后无法获取，条目计数减少
    #[tokio::test]
    async fn test_remove() {
        let slot_manager = create_test_slot_manager(4);
        let strategy = LruCacheStrategy::new(
            slot_manager,
            Duration::from_secs(60),
            10
        );

        // 插入条目
        strategy.insert("dialogue-1".to_string(), 0, 100).await;
        assert_eq!(strategy.entry_count().await, 1);

        // 删除条目
        strategy.remove("dialogue-1").await;
        
        // 验证条目已删除
        assert_eq!(strategy.entry_count().await, 0);
        assert!(strategy.get("dialogue-1").await.is_none());
    }

    /// 测试 LRU 驱逐策略
    /// 验证: 超过最大条目数时，最久未访问的条目被驱逐
    #[tokio::test]
    async fn test_lru_eviction() {
        let slot_manager = create_test_slot_manager(4);
        let strategy = LruCacheStrategy::new(
            slot_manager,
            Duration::from_millis(10),  // 短保留时间便于测试
            2                           // 最多 2 个条目
        );

        // 插入 3 个条目，超过最大限制
        strategy.insert("dialogue-1".to_string(), 0, 100).await;
        strategy.insert("dialogue-2".to_string(), 1, 200).await;
        
        // 等待 retention_duration，使前两个条目过期
        tokio::time::sleep(Duration::from_millis(20)).await;
        
        // 插入第三个条目，触发清理和驱逐
        strategy.insert("dialogue-3".to_string(), 2, 300).await;

        // 验证只保留 2 个条目
        assert_eq!(strategy.entry_count().await, 2);
        
        // dialogue-3 应该存在
        assert!(strategy.get("dialogue-3").await.is_some());
    }

    /// 测试访问后更新最后访问时间
    /// 验证: 获取条目后，last_accessed_at 被更新，且条目不会被立即驱逐
    #[tokio::test]
    async fn test_access_updates_timestamp() {
        let slot_manager = create_test_slot_manager(4);
        let strategy = LruCacheStrategy::new(
            slot_manager,
            Duration::from_millis(50),
            2
        );

        // 插入两个条目
        strategy.insert("dialogue-1".to_string(), 0, 100).await;
        strategy.insert("dialogue-2".to_string(), 1, 200).await;
        
        // 等待一段时间
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        // 访问 dialogue-1，更新其时间戳
        strategy.get("dialogue-1").await;
        
        // 再等待一段时间，使 dialogue-2 过期但 dialogue-1 不过期
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        // 插入新条目触发清理
        strategy.insert("dialogue-3".to_string(), 2, 300).await;
        
        // dialogue-1 应该保留（刚访问过）
        assert!(strategy.get("dialogue-1").await.is_some());
    }

    /// 测试批量插入
    /// 验证: 批量插入多个条目成功
    #[tokio::test]
    async fn test_insert_batch() {
        let slot_manager = create_test_slot_manager(4);
        let strategy = LruCacheStrategy::new(
            slot_manager,
            Duration::from_secs(60),
            10
        );

        let entries = vec![
            ("dialogue-1".to_string(), 0, 100),
            ("dialogue-2".to_string(), 1, 200),
            ("dialogue-3".to_string(), 2, 300),
        ];
        
        strategy.insert_batch(entries).await;
        
        assert_eq!(strategy.entry_count().await, 3);
        
        // 验证每个条目都能获取到
        assert!(strategy.get("dialogue-1").await.is_some());
        assert!(strategy.get("dialogue-2").await.is_some());
        assert!(strategy.get("dialogue-3").await.is_some());
    }

    /// 测试批量删除
    /// 验证: 批量删除多个条目成功
    #[tokio::test]
    async fn test_remove_batch() {
        let slot_manager = create_test_slot_manager(4);
        let strategy = LruCacheStrategy::new(
            slot_manager,
            Duration::from_secs(60),
            10
        );

        // 插入多个条目
        strategy.insert("dialogue-1".to_string(), 0, 100).await;
        strategy.insert("dialogue-2".to_string(), 1, 200).await;
        strategy.insert("dialogue-3".to_string(), 2, 300).await;
        
        assert_eq!(strategy.entry_count().await, 3);

        // 批量删除
        strategy.remove_batch(&["dialogue-1", "dialogue-3"]).await;
        
        assert_eq!(strategy.entry_count().await, 1);
        
        // 验证只有 dialogue-2 保留
        assert!(strategy.get("dialogue-1").await.is_none());
        assert!(strategy.get("dialogue-2").await.is_some());
        assert!(strategy.get("dialogue-3").await.is_none());
    }

    /// 测试清理过期条目
    /// 验证: cleanup 方法正确将过期条目加入 LRU 并驱逐
    #[tokio::test]
    async fn test_cleanup() {
        let slot_manager = create_test_slot_manager(4);
        let strategy = LruCacheStrategy::new(
            slot_manager,
            Duration::from_millis(20),
            3
        );

        // 插入多个条目
        strategy.insert("d1".to_string(), 0, 10).await;
        strategy.insert("d2".to_string(), 1, 20).await;
        strategy.insert("d3".to_string(), 2, 30).await;
        
        // 等待过期
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        // 手动触发清理
        strategy.cleanup(Instant::now()).await;
        
        // 插入新条目，应该触发驱逐
        strategy.insert("d4".to_string(), 3, 40).await;
        
        // 应该只有 3 个条目
        assert_eq!(strategy.entry_count().await, 3);
    }

    /// 测试更新已存在的条目
    /// 验证: 插入相同 dialogue_id 会更新条目
    #[tokio::test]
    async fn test_insert_updates_existing() {
        let slot_manager = create_test_slot_manager(4);
        let strategy = LruCacheStrategy::new(
            slot_manager,
            Duration::from_secs(60),
            10
        );

        // 插入条目
        strategy.insert("dialogue-1".to_string(), 0, 100).await;
        
        let entry = strategy.get("dialogue-1").await.unwrap();
        assert_eq!(entry.token_count, 100);
        assert_eq!(entry.slot_index, 0);

        // 使用相同 dialogue_id 插入新数据
        strategy.insert("dialogue-1".to_string(), 1, 200).await;
        
        let entry = strategy.get("dialogue-1").await.unwrap();
        assert_eq!(entry.token_count, 200);
        assert_eq!(entry.slot_index, 1);
        
        // 仍然只有一个条目
        assert_eq!(strategy.entry_count().await, 1);
    }
}
