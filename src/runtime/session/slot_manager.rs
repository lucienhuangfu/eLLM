use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;
use tokio::sync::Mutex as TokioMutex;

use super::slot_state::SlotState;
use super::types::{SessionHandle, SessionMode, SlotError};
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::state::batch::BatchSequence;

const LRU_SENTINEL: usize = usize::MAX;

/// Session 元数据（由 SlotManager 统一管理，不再嵌入 SlotState）
struct SessionMeta {
    token_count: usize,
}

/// SlotManager — 统一管理 slot 的执行状态、LRU 调度和 session 生命周期。
///
/// 拥有 `SharedMut<Vec<SlotState>>` 作为唯一的 slot 状态数据源，
/// Scheduler 和 serving 层通过 `batch_list()` / `with_slots()` 共享访问。
pub struct SlotManager<T: Copy + crate::num_traits::FromNumber> {
    /// 唯一的 slot 状态存储 —— Scheduler / serving 共享
    batch_states: Arc<SharedMut<Vec<SlotState>>>,
    /// batch_sequences 用于 tokenizer / delta 计算
    batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    /// LRU 双向链表（独立数组，StdMutex 保护）
    lru: StdMutex<LruLists>,
    /// session_id → slot_index 活跃映射
    session_map: TokioMutex<HashMap<String, usize>>,
    /// session_id → (slot_index, cancel_flag) 延迟回收池
    reserved_slots: Arc<TokioMutex<HashMap<String, (usize, Arc<AtomicBool>)>>>,
    /// 每个 slot 的 session 元数据
    meta: StdMutex<Vec<SessionMeta>>,
    mode: SessionMode,
    reuse_timeout: Duration,
}

/// LRU 双向链表数组
struct LruLists {
    prev: Vec<usize>,
    next: Vec<usize>,
}

impl LruLists {
    fn new(n: usize) -> Self {
        let (mut prev, mut next) = (vec![LRU_SENTINEL; n], vec![LRU_SENTINEL; n]);
        if n > 1 {
            prev[0] = n - 1;
            next[0] = 1;
            for i in 1..n - 1 {
                prev[i] = i - 1;
                next[i] = i + 1;
            }
            prev[n - 1] = n - 2;
        }
        Self { prev, next }
    }

    fn touch(&mut self, idx: usize) {
        let p = self.prev[idx];
        let n = self.next[idx];
        if p != LRU_SENTINEL {
            self.next[p] = n;
        }
        if n != LRU_SENTINEL {
            self.prev[n] = p;
        }
        let head_prev = self.prev[0];
        self.prev[idx] = LRU_SENTINEL;
        self.next[idx] = head_prev;
        if head_prev != LRU_SENTINEL {
            self.next[head_prev] = idx;
        }
        self.prev[0] = idx;
    }

    fn insert_head(&mut self, idx: usize) {
        let head_prev = self.prev[0];
        self.prev[idx] = LRU_SENTINEL;
        self.next[idx] = head_prev;
        if head_prev != LRU_SENTINEL {
            self.next[head_prev] = idx;
        }
        self.prev[0] = idx;
    }

    fn evict_tail(&mut self) -> usize {
        let mut tail = 0;
        while self.next[tail] != LRU_SENTINEL {
            tail = self.next[tail];
        }
        let p = self.prev[tail];
        if p != LRU_SENTINEL {
            self.next[p] = LRU_SENTINEL;
        }
        self.prev[tail] = LRU_SENTINEL;
        self.next[tail] = LRU_SENTINEL;
        tail
    }

    fn detach(&mut self, idx: usize) {
        let p = self.prev[idx];
        let n = self.next[idx];
        if p != LRU_SENTINEL {
            self.next[p] = n;
        }
        if n != LRU_SENTINEL {
            self.prev[n] = p;
        }
        self.prev[idx] = LRU_SENTINEL;
        self.next[idx] = LRU_SENTINEL;
    }
}

impl<T: Copy + crate::num_traits::FromNumber> SlotManager<T> {
    pub fn new(
        num_slots: usize,
        batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
        batch_states: Arc<SharedMut<Vec<SlotState>>>,
        mode: SessionMode,
        reuse_timeout_ms: u64,
    ) -> Self {
        let meta = (0..num_slots)
            .map(|_| SessionMeta { token_count: 0 })
            .collect();

        Self {
            batch_states,
            batch_sequences,
            lru: StdMutex::new(LruLists::new(num_slots)),
            session_map: TokioMutex::new(HashMap::new()),
            reserved_slots: Arc::new(TokioMutex::new(HashMap::new())),
            meta: StdMutex::new(meta),
            mode,
            reuse_timeout: Duration::from_millis(reuse_timeout_ms),
        }
    }

    // ── 公共访问接口 ─────────────────────────────────────────

    /// 获取 batch_states 的 Arc 克隆（供 Scheduler 等外部使用）
    pub fn batch_list(&self) -> Arc<SharedMut<Vec<SlotState>>> {
        Arc::clone(&self.batch_states)
    }

    /// 只读访问 slot 状态
    pub fn with_slots<R>(&self, f: impl FnOnce(&[SlotState]) -> R) -> R {
        self.batch_states.with(|v| f(v.as_slice()))
    }

    /// 可变访问 slot 状态
    pub fn with_slots_mut<R>(&self, f: impl FnOnce(&mut [SlotState]) -> R) -> R {
        self.batch_states.with_mut(|v| f(v.as_mut_slice()))
    }

    /// 将 slot 从 LRU 中摘除（开始写入 prompt 时调用）
    pub fn detach_from_lru(&self, idx: usize) {
        self.lru.lock().unwrap().detach(idx);
    }

    // ── Session 生命周期 ─────────────────────────────────────

    pub async fn acquire_session(&self, session_id: &str) -> Result<SessionHandle, SlotError> {
        let mut map = self.session_map.lock().await;

        // 1) 活跃会话映射中查找
        if let Some(&slot_index) = map.get(session_id) {
            return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
        }

        // 2) 保留池中查找 — 取消计时器并复用
        let reserved = {
            let mut r = self.reserved_slots.lock().await;
            r.remove(session_id)
        };

        if let Some((slot_index, cancel_flag)) = reserved {
            cancel_flag.store(true, Ordering::Release);
            map.insert(session_id.to_string(), slot_index);
            return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
        }

        // 3) 从 LRU 尾部取最久未使用的 slot
        let slot_index = self.lru.lock().unwrap().evict_tail();

        // 清理可能残留的旧 session 映射
        if let Some(old_id) = map
            .iter()
            .find(|(_, &idx)| idx == slot_index)
            .map(|(k, _)| k.clone())
        {
            map.remove(&old_id);
        }
        map.insert(session_id.to_string(), slot_index);

        // 重置 slot 执行状态
        self.batch_states.with_mut(|slots| {
            slots[slot_index] = SlotState::new_start_state();
        });

        Ok(SessionHandle::new(session_id.to_string(), slot_index))
    }

    pub async fn release_session(&self, session_id: &str, token_count: usize) {
        let mut map = self.session_map.lock().await;
        let Some(&slot_index) = map.get(session_id) else {
            return;
        };

        self.meta.lock().unwrap()[slot_index].token_count = token_count;

        if self.mode == SessionMode::NonReusable {
            self.batch_states.with_mut(|slots| {
                slots[slot_index].reset_to_start();
            });
            map.remove(session_id);
            self.lru.lock().unwrap().insert_head(slot_index);
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
        let batch_states = Arc::clone(&self.batch_states);
        let lru = &self.lru as *const StdMutex<LruLists> as usize;
        let timeout = self.reuse_timeout;

        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            if cancel_flag.load(Ordering::Acquire) {
                return;
            }

            // SAFETY: SlotManager 由 Arc 持有，spawn 期间不会被释放
            let mut reserved = reserved_slots.lock().await;
            if let Some((idx, _)) = reserved.remove(&session_id_owned) {
                batch_states.with_mut(|slots| {
                    slots[idx].reset_to_start();
                });
                let lru = unsafe { &*(lru as *const StdMutex<LruLists>) };
                lru.lock().unwrap().insert_head(idx);
            }
        });
    }

    pub async fn calculate_delta(
        &self,
        session_id: &str,
        new_tokens: &[u32],
    ) -> Option<(usize, Vec<u32>)> {
        let map = self.session_map.lock().await;
        let &slot_index = map.get(session_id)?;
        let cached_count = self.meta.lock().unwrap()[slot_index].token_count;
        if cached_count == 0 {
            return None;
        }
        drop(map);

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

    fn model_dir() -> String {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("models");
        p.push("MiniMax-M2.5");
        p.to_string_lossy().into_owned()
    }

    fn create_test_manager(batch_size: usize, timeout_ms: u64) -> Arc<SlotManager<f16>> {
        let dir = model_dir();
        let batch_sequences = Arc::new(SharedMut::new(
            BatchSequence::<f16>::new(
                std::ptr::null_mut(),
                batch_size,
                1024,
                &format!("{}/tokenizer.json", dir),
                &format!("{}/tokenizer_config.json", dir),
                &format!("{}/chat_template.jinja", dir),
            )
            .unwrap(),
        ));
        let batch_states = Arc::new(SharedMut::new(
            (0..batch_size)
                .map(|_| SlotState::new_start_state())
                .collect::<Vec<_>>(),
        ));
        Arc::new(SlotManager::new(
            batch_size,
            batch_sequences,
            batch_states,
            SessionMode::Reusable,
            timeout_ms,
        ))
    }

    #[tokio::test]
    async fn test_slot_reserved_and_reused() {
        let manager = create_test_manager(4, 1000);

        let handle1 = manager.acquire_session("session1").await.unwrap();
        assert!(!handle1.is_reused);

        manager.release_session("session1", 10).await;

        let handle2 = manager.acquire_session("session1").await.unwrap();
        assert_eq!(handle2.slot_index, handle1.slot_index);
        assert!(handle2.is_reused);
    }

    #[tokio::test]
    async fn test_slot_timeout_to_lru() {
        let manager = create_test_manager(4, 300);

        manager.acquire_session("session1").await.unwrap();
        manager.release_session("session1", 10).await;

        tokio::time::sleep(Duration::from_millis(400)).await;

        let h2 = manager.acquire_session("s2").await.unwrap();
        let h3 = manager.acquire_session("s3").await.unwrap();
        let h4 = manager.acquire_session("s4").await.unwrap();
        let h5 = manager.acquire_session("s5").await.unwrap();
        assert!(!h2.is_reused);
        assert!(!h3.is_reused);
        assert!(!h4.is_reused);
        assert!(!h5.is_reused);
    }

    #[tokio::test]
    async fn test_non_reusable_mode_immediate_release() {
        let dir = model_dir();
        let batch_sequences = Arc::new(SharedMut::new(
            BatchSequence::<f16>::new(
                std::ptr::null_mut(),
                4,
                1024,
                &format!("{}/tokenizer.json", dir),
                &format!("{}/tokenizer_config.json", dir),
                &format!("{}/chat_template.jinja", dir),
            )
            .unwrap(),
        ));
        let batch_states = Arc::new(SharedMut::new(
            (0..4)
                .map(|_| SlotState::new_start_state())
                .collect::<Vec<_>>(),
        ));
        let manager = Arc::new(SlotManager::new(
            4,
            batch_sequences,
            batch_states,
            SessionMode::NonReusable,
            1000,
        ));

        manager.acquire_session("session1").await.unwrap();
        manager.release_session("session1", 10).await;

        let handle2 = manager.acquire_session("session2").await.unwrap();
        assert!(!handle2.is_reused);
    }

    #[tokio::test]
    async fn test_with_slots_access() {
        let manager = create_test_manager(4, 1000);

        let all_start = manager.with_slots(|slots| slots.iter().all(|s| s.is_available()));
        assert!(all_start);

        let h = manager.acquire_session("s1").await.unwrap();
        let phase = manager.with_slots(|slots| slots[h.slot_index].phase);
        assert_eq!(phase, crate::runtime::session::Phase::Start);
    }
}
