use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant};
use tokio::sync::Mutex as TokioMutex;

use super::slot_state::SlotState;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::state::batch::BatchSequence;

use super::types::{SessionHandle, SessionMode, SlotError};

const LRU_SENTINEL: usize = usize::MAX;

pub struct SlotManager<T: Copy + crate::num_traits::FromNumber> {
    slots: Arc<StdMutex<Vec<SlotState>>>,
    available_slots: Arc<TokioMutex<Vec<usize>>>,
    session_map: Arc<TokioMutex<HashMap<String, usize>>>,
    reserved_slots: Arc<TokioMutex<HashMap<String, (usize, Arc<AtomicBool>)>>>,
    batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    mode: SessionMode,
    reuse_timeout: Duration,
}

impl<T: Copy + crate::num_traits::FromNumber> SlotManager<T> {
    pub fn new(
        num_slots: usize,
        batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
        mode: SessionMode,
        reuse_timeout_ms: u64,
    ) -> Self {
        let slots: Vec<_> = (0..num_slots)
            .map(|_| SlotState::new_start_state())
            .collect();
        let available_slots: Vec<_> = (0..num_slots).collect();

        Self {
            slots: Arc::new(StdMutex::new(slots)),
            available_slots: Arc::new(TokioMutex::new(available_slots)),
            session_map: Arc::new(TokioMutex::new(HashMap::new())),
            reserved_slots: Arc::new(TokioMutex::new(HashMap::new())),
            batch_sequences,
            mode,
            reuse_timeout: Duration::from_millis(reuse_timeout_ms),
        }
    }

    /// 将指定 slot 移到 LRU 链表头部（最近使用）
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

    /// 驱逐 LRU 链表尾部（最久未使用）的 slot
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
        // 1) 活跃会话映射中查找
        let slot_from_map = {
            let map = self.session_map.lock().await;
            map.get(session_id).copied()
        };

        if let Some(slot_index) = slot_from_map {
            let mut slots = self.slots.lock().unwrap();
            if let Some(entry) = slots.get_mut(slot_index) {
                entry.touch();
                drop(slots);
                self.touch_lru(slot_index);
                return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
            }
        }

        // 2) 保留池中查找 — 取消计时器并复用
        let reserved = {
            let mut r = self.reserved_slots.lock().await;
            r.remove(session_id)
        };

        if let Some((slot_index, cancel_flag)) = reserved {
            cancel_flag.store(true, Ordering::Release);
            {
                let mut map = self.session_map.lock().await;
                map.insert(session_id.to_string(), slot_index);
            }
            let mut slots = self.slots.lock().unwrap();
            if let Some(entry) = slots.get_mut(slot_index) {
                entry.touch();
                drop(slots);
                self.touch_lru(slot_index);
                return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
            }
        }

        // 3) 分配新槽位
        let slot_index = {
            let mut avail = self.available_slots.lock().await;
            avail.pop().unwrap_or_else(|| self.evict_oldest())
        };

        {
            let mut map = self.session_map.lock().await;
            if let Some(old_id) = map
                .iter()
                .find(|(_, &idx)| idx == slot_index)
                .map(|(k, _)| k.clone())
            {
                map.remove(&old_id);
            }
            map.insert(session_id.to_string(), slot_index);
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
        let mut map = self.session_map.lock().await;
        let Some(&slot_index) = map.get(session_id) else {
            return;
        };

        {
            let mut slots = self.slots.lock().unwrap();
            if let Some(entry) = slots.get_mut(slot_index) {
                entry.token_count = token_count;
            }
        }

        if self.mode == SessionMode::NonReusable {
            {
                let mut slots = self.slots.lock().unwrap();
                if let Some(entry) = slots.get_mut(slot_index) {
                    entry.reset_to_start();
                }
            }
            map.remove(session_id);
            let mut avail = self.available_slots.lock().await;
            avail.push(slot_index);
            return;
        }

        // Reusable: 启动延迟回收定时器
        let session_id_owned = session_id.to_string();
        map.remove(session_id);

        let cancel_flag = Arc::new(AtomicBool::new(false));
        {
            let mut reserved = self.reserved_slots.lock().await;
            reserved.insert(
                session_id_owned.clone(),
                (slot_index, Arc::clone(&cancel_flag)),
            );
        }

        let reserved_slots = Arc::clone(&self.reserved_slots);
        let available_slots = Arc::clone(&self.available_slots);
        let slots = Arc::clone(&self.slots);
        let timeout = self.reuse_timeout;

        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            if cancel_flag.load(Ordering::Acquire) {
                return;
            }

            let mut reserved = reserved_slots.lock().await;
            if let Some((idx, _)) = reserved.remove(&session_id_owned) {
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
                }

                let mut avail = available_slots.lock().await;
                if !avail.contains(&idx) {
                    avail.push(idx);
                }
            }
        });
    }

    pub async fn remove_from_available(&self, slot_index: usize) {
        let mut avail = self.available_slots.lock().await;
        if let Some(pos) = avail.iter().position(|&idx| idx == slot_index) {
            avail.swap_remove(pos);
        }
    }

    pub async fn calculate_delta(
        &self,
        session_id: &str,
        new_tokens: &[u32],
    ) -> Option<(usize, Vec<u32>)> {
        let (slot_index, cached_count) = {
            let map = self.session_map.lock().await;
            let &slot_index = map.get(session_id)?;
            let slots = self.slots.lock().unwrap();
            let entry = slots.get(slot_index)?;
            if entry.token_count > 0 {
                (slot_index, entry.token_count)
            } else {
                return None;
            }
        };

        let cached_tokens = self
            .batch_sequences
            .with(|batch_seq| batch_seq.token_ids(slot_index, 0, cached_count));

        let min_len = cached_tokens.len().min(new_tokens.len());
        let prefix_len = (0..min_len)
            .take_while(|&i| cached_tokens[i] == new_tokens[i])
            .count();

        if prefix_len > 0 {
            Some((prefix_len, new_tokens[prefix_len..].to_vec()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::state::batch::BatchSequence;
    use std::time::Duration;

    fn create_test_manager(batch_size: usize, timeout_ms: u64) -> Arc<SlotManager<f16>> {
        let batch_sequences = Arc::new(SharedMut::new(
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
        let manager = create_test_manager(4, 1000);

        let handle1 = manager.acquire_session("session1").await.unwrap();
        assert_eq!(handle1.slot_index, 0);
        assert!(!handle1.is_reused);

        manager.release_session("session1", 10).await;

        let handle2 = manager.acquire_session("session1").await.unwrap();
        assert_eq!(handle2.slot_index, 0);
        assert!(handle2.is_reused);

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_slot_timeout_release() {
        let manager = create_test_manager(4, 500);

        let handle1 = manager.acquire_session("session1").await.unwrap();
        let slot_idx = handle1.slot_index;
        manager.release_session("session1", 10).await;

        tokio::time::sleep(Duration::from_millis(600)).await;

        let handle2 = manager.acquire_session("session2").await.unwrap();
        println!(
            "Session1 slot {}, Session2 slot {}",
            slot_idx, handle2.slot_index
        );
    }

    #[tokio::test]
    async fn test_non_reusable_mode_immediate_release() {
        let batch_sequences = Arc::new(SharedMut::new(
            BatchSequence::<f16>::new(std::ptr::null_mut(), 4, 1024, "gpt2", "gpt2", "gpt2")
                .unwrap(),
        ));
        let manager = Arc::new(SlotManager::new(
            4,
            batch_sequences,
            SessionMode::NonReusable,
            1000,
        ));

        let handle1 = manager.acquire_session("session1").await.unwrap();
        let slot_idx = handle1.slot_index;
        manager.release_session("session1", 10).await;

        let handle2 = manager.acquire_session("session2").await.unwrap();
        println!(
            "NonReusable: Session1={}, Session2={}",
            slot_idx, handle2.slot_index
        );
    }
}
