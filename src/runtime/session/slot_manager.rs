use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant};
use tokio::sync::Mutex as TokioMutex;

use crate::runtime::error::SlotError;
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::core::SlotState;
use crate::runtime::state::types::Phase;

use super::types::{SessionHandle, SessionMode};

const LRU_SENTINEL: usize = usize::MAX;

pub struct SlotManager<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    slots: Arc<StdMutex<Vec<SlotState>>>,
    available_slots: Arc<TokioMutex<Vec<usize>>>,
    session_map: Arc<TokioMutex<HashMap<String, usize>>>,
    reserved_slots: Arc<TokioMutex<HashMap<String, (usize, Arc<AtomicBool>)>>>, // session_id -> (slot_index, cancel_flag)
    batch_sequences: Arc<crate::operators::send_sync_ptr::SharedMut<BatchSequence<T>>>,
    mode: SessionMode,
    reuse_timeout: Duration,
}

unsafe impl<T> Send for SlotManager<T> where T: Copy + crate::num_traits::FromNumber + Send {}

unsafe impl<T> Sync for SlotManager<T> where T: Copy + crate::num_traits::FromNumber + Sync {}

impl<T> SlotManager<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    pub fn new(
        num_slots: usize,
        batch_sequences: Arc<crate::operators::send_sync_ptr::SharedMut<BatchSequence<T>>>,
        mode: SessionMode,
        reuse_timeout_ms: u64,
    ) -> Self {
        let mut slots = Vec::with_capacity(num_slots);
        let mut available_slots = Vec::with_capacity(num_slots);

        for i in 0..num_slots {
            slots.push(SlotState::new_start_state());
            available_slots.push(i);
        }

        let mut slot_manager = Self {
            slots: Arc::new(StdMutex::new(slots)),
            available_slots: Arc::new(TokioMutex::new(available_slots)),
            session_map: Arc::new(TokioMutex::new(HashMap::new())),
            reserved_slots: Arc::new(TokioMutex::new(HashMap::new())),
            batch_sequences,
            mode,
            reuse_timeout: Duration::from_millis(reuse_timeout_ms),
        };

        slot_manager
    }

    fn init_lru(&mut self) {
        let mut slots = self.slots.lock().unwrap();
        let num_slots = slots.len();

        for i in 0..num_slots {
            slots[i].lru_prev = if i == 0 { LRU_SENTINEL } else { i - 1 };
            slots[i].lru_next = if i == num_slots - 1 {
                LRU_SENTINEL
            } else {
                i + 1
            };
        }
    }

    fn touch_lru(&self, slot_index: usize) {
        let mut slots = self.slots.lock().unwrap();

        let prev = slots[slot_index].lru_prev;
        let next = slots[slot_index].lru_next;

        if prev != LRU_SENTINEL {
            slots[prev].lru_next = next;
        }
        if next != LRU_SENTINEL {
            slots[next].lru_prev = prev;
        }

        let head_prev = slots[0].lru_prev;

        slots[slot_index].lru_prev = LRU_SENTINEL;
        slots[slot_index].lru_next = head_prev;

        if head_prev != LRU_SENTINEL {
            slots[head_prev].lru_next = slot_index;
        }
        slots[0].lru_prev = slot_index;
    }

    fn evict_oldest(&self) -> usize {
        let mut slots = self.slots.lock().unwrap();

        let mut tail = 0;
        while slots[tail].lru_next != LRU_SENTINEL {
            tail = slots[tail].lru_next;
        }

        let prev = slots[tail].lru_prev;
        if prev != LRU_SENTINEL {
            slots[prev].lru_next = LRU_SENTINEL;
        }

        tail
    }

    pub async fn acquire_session(&self, session_id: &str) -> Result<SessionHandle, SlotError> {
        // 首先检查是否有活跃的会话映射
        let slot_index_from_map = {
            let session_map = self.session_map.lock().await;
            session_map.get(session_id).copied()
        };

        if let Some(slot_index) = slot_index_from_map {
            let mut slots = self.slots.lock().unwrap();
            if let Some(entry) = slots.get_mut(slot_index) {
                entry.touch();
                self.touch_lru(slot_index);
                return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
            }
        }

        // 检查是否有保留的槽位（reserved_slots）
        let reserved_result = {
            let mut reserved = self.reserved_slots.lock().await;
            reserved.remove(session_id)
        };

        if let Some((slot_index, cancel_flag)) = reserved_result {
            // 取消计时器
            cancel_flag.store(true, Ordering::Release);

            // 恢复会话映射
            {
                let mut session_map = self.session_map.lock().await;
                session_map.insert(session_id.to_string(), slot_index);
            }

            let mut slots = self.slots.lock().unwrap();
            if let Some(entry) = slots.get_mut(slot_index) {
                entry.touch();
                self.touch_lru(slot_index);
                return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
            }
        }

        // 分配新槽位
        let slot_index = {
            let mut available = self.available_slots.lock().await;
            if !available.is_empty() {
                available.pop().unwrap()
            } else {
                self.evict_oldest()
            }
        };

        {
            let mut session_map = self.session_map.lock().await;
            let old_session_id: Option<String> = session_map
                .iter()
                .find(|(_, &idx)| idx == slot_index)
                .map(|(k, _)| k.clone());
            if let Some(id) = old_session_id {
                session_map.remove(&id);
            }
            session_map.insert(session_id.to_string(), slot_index);
        }

        {
            let mut slots = self.slots.lock().unwrap();
            let entry = &mut slots[slot_index];
            entry.session_id = Some(session_id.to_string());
            entry.created_at = Instant::now();
            entry.last_accessed = Instant::now();
            entry.token_count = 0;
        }

        self.touch_lru(slot_index);

        Ok(SessionHandle::new(session_id.to_string(), slot_index))
    }

    pub async fn release_session(&self, session_id: &str, token_count: usize) {
        let mut session_map = self.session_map.lock().await;
        if let Some(&slot_index) = session_map.get(session_id) {
            {
                let mut slots = self.slots.lock().unwrap();
                if let Some(entry) = slots.get_mut(slot_index) {
                    entry.token_count = token_count;
                }
            } // Release slots lock before await

            if self.mode == SessionMode::NonReusable {
                // NonReusable 模式：立即重置并释放
                {
                    let mut slots = self.slots.lock().unwrap();
                    if let Some(entry) = slots.get_mut(slot_index) {
                        entry.reset_to_start();
                    }
                }
                session_map.remove(session_id);

                let mut available = self.available_slots.lock().await;
                available.push(slot_index);
            } else {
                // Reusable 模式：启动延迟回收定时器，超时后进入LRU队列
                let session_id_owned = session_id.to_string();
                session_map.remove(session_id);

                // 创建取消标志
                let cancel_flag = Arc::new(AtomicBool::new(false));

                // 添加到保留列表
                {
                    let mut reserved = self.reserved_slots.lock().await;
                    reserved.insert(
                        session_id_owned.clone(),
                        (slot_index, Arc::clone(&cancel_flag)),
                    );
                }

                // 启动异步计时器
                let reserved_slots = Arc::clone(&self.reserved_slots);
                let available_slots = Arc::clone(&self.available_slots);
                let slots = Arc::clone(&self.slots);
                let timeout = self.reuse_timeout;

                tokio::spawn(async move {
                    tokio::time::sleep(timeout).await;

                    // 检查是否被取消
                    if cancel_flag.load(Ordering::Acquire) {
                        return; // 已被复用，不执行回收
                    }

                    // 超时后从 reserved 移除，更新LRU并加入可用池
                    let mut reserved = reserved_slots.lock().await;
                    if let Some((idx, _)) = reserved.remove(&session_id_owned) {
                        // 更新LRU，将slot放入LRU链表头部（表示最近使用）
                        {
                            let mut slots = slots.lock().unwrap();
                            let prev = slots[idx].lru_prev;
                            let next = slots[idx].lru_next;
                            if prev != LRU_SENTINEL {
                                slots[prev].lru_next = next;
                            }
                            if next != LRU_SENTINEL {
                                slots[next].lru_prev = prev;
                            }
                            let head_prev = slots[0].lru_prev;
                            slots[idx].lru_prev = LRU_SENTINEL;
                            slots[idx].lru_next = head_prev;
                            if head_prev != LRU_SENTINEL {
                                slots[head_prev].lru_next = idx;
                            }
                            slots[0].lru_prev = idx;
                        } // Release slots lock before await

                        let mut available = available_slots.lock().await;
                        if !available.contains(&idx) {
                            available.push(idx);
                        }
                    }
                });
            }
        }
    }

    pub async fn remove_from_available(&self, slot_index: usize) {
        let mut available = self.available_slots.lock().await;
        if let Some(pos) = available.iter().position(|&idx| idx == slot_index) {
            available.swap_remove(pos);
        }
    }

    pub async fn get_slot(&self, slot_index: usize) -> Option<SlotState> {
        let slots = self.slots.lock().unwrap();
        slots.get(slot_index).cloned()
    }

    pub async fn get_cached_tokens(&self, session_id: &str) -> Option<(usize, usize)> {
        let session_map = self.session_map.lock().await;
        let &slot_index = session_map.get(session_id)?;
        let slots = self.slots.lock().unwrap();
        let entry = slots.get(slot_index)?;
        if entry.token_count > 0 {
            Some((slot_index, entry.token_count))
        } else {
            None
        }
    }

    pub async fn calculate_delta(
        &self,
        session_id: &str,
        new_tokens: &[u32],
    ) -> Option<(usize, Vec<u32>)> {
        let (slot_index, cached_count) = self.get_cached_tokens(session_id).await?;

        let cached_tokens = self
            .batch_sequences
            .with(|batch_seq| batch_seq.token_ids(slot_index, 0, cached_count));

        let min_len = cached_tokens.len().min(new_tokens.len());
        let mut prefix_len = 0;

        while prefix_len < min_len && cached_tokens[prefix_len] == new_tokens[prefix_len] {
            prefix_len += 1;
        }

        if prefix_len > 0 {
            Some((prefix_len, new_tokens[prefix_len..].to_vec()))
        } else {
            None
        }
    }

    pub async fn session_count(&self) -> usize {
        self.session_map.lock().await.len()
    }

    pub fn total_slots(&self) -> usize {
        self.slots.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::state::batch::BatchSequence;
    use std::time::Duration;

    fn create_test_manager(batch_size: usize, timeout_ms: u64) -> Arc<SlotManager<f16>> {
        let batch_sequences = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(
            BatchSequence::<f16>::new(
                std::ptr::null_mut(),
                batch_size,
                1024,
                "gpt2",
                "gpt2",
                "gpt2",
            )
            .unwrap(),
        ));
        Arc::new(SlotManager::new(
            batch_size,
            batch_sequences,
            SessionMode::Reusable,
            timeout_ms,
        ))
    }

    #[tokio::test]
    async fn test_slot_reserved_and_reused() {
        let manager = create_test_manager(4, 1000); // 1 second timeout

        // Acquire session
        let handle1 = manager.acquire_session("session1").await.unwrap();
        assert_eq!(handle1.slot_index, 0);
        assert!(!handle1.is_reused);

        // Release session (should start timer)
        manager.release_session("session1", 10).await;

        // Immediately acquire again - should reuse the reserved slot
        let handle2 = manager.acquire_session("session1").await.unwrap();
        assert_eq!(handle2.slot_index, 0);
        assert!(handle2.is_reused); // Should be reused from reserved

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_slot_timeout_release() {
        let manager = create_test_manager(4, 500); // 500ms timeout

        // Acquire and release
        let handle1 = manager.acquire_session("session1").await.unwrap();
        let slot_idx = handle1.slot_index;
        manager.release_session("session1", 10).await;

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(600)).await;

        // Now another session should be able to use this slot
        let handle2 = manager.acquire_session("session2").await.unwrap();
        // The slot should eventually become available (might not be the same index due to LRU)

        println!(
            "Session1 used slot {}, Session2 used slot {}",
            slot_idx, handle2.slot_index
        );
    }

    #[tokio::test]
    async fn test_non_reusable_mode_immediate_release() {
        let batch_sequences = Arc::new(crate::operators::send_sync_ptr::SharedMut::new(
            BatchSequence::<f16>::new(std::ptr::null_mut(), 4, 1024, "gpt2", "gpt2", "gpt2")
                .unwrap(),
        ));
        let manager = Arc::new(SlotManager::new(
            4,
            batch_sequences,
            SessionMode::NonReusable,
            1000,
        ));

        // Acquire and release in NonReusable mode
        let handle1 = manager.acquire_session("session1").await.unwrap();
        let slot_idx = handle1.slot_index;
        manager.release_session("session1", 10).await;

        // In NonReusable mode, slot should be immediately available
        let handle2 = manager.acquire_session("session2").await.unwrap();
        // Should get a slot immediately (might be the same or different)

        println!(
            "NonReusable: Session1={}, Session2={}",
            slot_idx, handle2.slot_index
        );
    }
}
