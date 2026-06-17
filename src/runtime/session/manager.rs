use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::SlotResult;
use crate::runtime::session::allocator::SlotAllocator;
use crate::runtime::session::types::{DialogueSession, SessionHandle, SessionMode};
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::types::SequenceState;

/// 会话管理器
///
/// 负责管理对话会话的生命周期，包括创建、复用、驱逐等
pub struct SessionManager<T> {
    sessions: Arc<Mutex<HashMap<String, DialogueSession>>>,
    slot_allocator: Arc<SlotAllocator>,
    batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    mode: SessionMode,
    slot_to_session: Arc<Mutex<Vec<Option<String>>>>,
}

impl<T> SessionManager<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    /// 创建新的会话管理器
    pub fn new(
        batch_states: Arc<SharedMut<Vec<SequenceState>>>,
        batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
        mode: SessionMode,
    ) -> Self {
        let slot_allocator = Arc::new(SlotAllocator::new(batch_states.clone()));
        let num_slots = batch_states.with(|states| states.len());
        let slot_to_session = Arc::new(Mutex::new(vec![None; num_slots]));

        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            slot_allocator,
            batch_sequences,
            mode,
            slot_to_session,
        }
    }

    /// 获取会话
    ///
    /// 如果会话已存在，则复用；否则创建新会话
    /// 时间复杂度:
    /// - 复用: O(1)
    /// - 创建: O(n) 分配 + O(1) 驱逐
    pub async fn acquire_session(&self, session_id: &str) -> SlotResult<SessionHandle> {
        let mut sessions = self.sessions.lock().await;

        // 复用已有会话
        if let Some(session) = sessions.get_mut(session_id) {
            session.touch();
            self.slot_allocator.touch(session.slot_index);
            return Ok(SessionHandle::reused(
                session_id.to_string(),
                session.slot_index,
            ));
        }

        // 分配槽位
        let slot_index = self.slot_allocator.allocate();

        // O(1) 驱逐旧会话
        let mut slot_to_session = self.slot_to_session.lock().await;
        let old_session_id = slot_to_session[slot_index].clone();

        if let Some(old_id) = old_session_id {
            sessions.remove(&old_id);
        }

        // O(1) 更新映射
        slot_to_session[slot_index] = Some(session_id.to_string());
        drop(slot_to_session);

        // 释放槽位
        self.slot_allocator.release(slot_index);
        self.slot_allocator.touch(slot_index);

        // 创建新会话
        let new_session = DialogueSession {
            session_id: session_id.to_string(),
            slot_index,
            token_count: 0,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
        };

        sessions.insert(session_id.to_string(), new_session);
        Ok(SessionHandle::new(session_id.to_string(), slot_index))
    }

    /// 释放会话
    pub async fn release_session(&self, session_id: &str, token_count: usize) {
        let mut sessions = self.sessions.lock().await;

        if let Some(session) = sessions.get_mut(session_id) {
            session.token_count = token_count;

            if self.mode == SessionMode::NonReusable {
                let slot_index = session.slot_index;
                sessions.remove(session_id);

                // 清除映射
                let mut slot_to_session = self.slot_to_session.lock().await;
                if slot_index < slot_to_session.len() {
                    slot_to_session[slot_index] = None;
                }
            }
        }
    }

    /// 获取缓存的 token
    pub async fn get_cached_tokens(&self, session_id: &str) -> Option<(usize, usize)> {
        let sessions = self.sessions.lock().await;
        sessions
            .get(session_id)
            .filter(|s| s.token_count > 0)
            .map(|s| (s.slot_index, s.token_count))
    }

    /// 计算 token delta
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

    /// 获取会话数量
    pub async fn session_count(&self) -> usize {
        let sessions = self.sessions.lock().await;
        sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_batch_states() -> Arc<SharedMut<Vec<SequenceState>>> {
        Arc::new(SharedMut::new(vec![SequenceState::new_start_state(); 4]))
    }

    fn create_test_batch_sequence() -> Arc<SharedMut<BatchSequence<f16>>> {
        Arc::new(SharedMut::new(BatchSequence::<f16>::default()))
    }

    #[tokio::test]
    async fn test_reusable_mode() {
        let batch_states = create_test_batch_states();
        let batch_seq = create_test_batch_sequence();
        let manager = SessionManager::new(batch_states, batch_seq, SessionMode::Reusable);

        let handle1 = manager.acquire_session("test-session").await.unwrap();
        assert_eq!(handle1.session_id, "test-session");
        assert!(!handle1.is_reused);

        let handle2 = manager.acquire_session("test-session").await.unwrap();
        assert_eq!(handle2.slot_index, handle1.slot_index);
        assert!(handle2.is_reused);
    }

    #[tokio::test]
    async fn test_non_reusable_mode() {
        let batch_states = create_test_batch_states();
        let batch_seq = create_test_batch_sequence();
        let manager = SessionManager::new(batch_states, batch_seq, SessionMode::NonReusable);

        let handle1 = manager.acquire_session("test-session").await.unwrap();
        let slot1 = handle1.slot_index;

        manager.release_session("test-session", 10).await;

        let handle2 = manager.acquire_session("test-session").await.unwrap();
        assert_ne!(handle2.slot_index, slot1);
        assert!(!handle2.is_reused);
    }

    #[tokio::test]
    async fn test_evict_oldest_session() {
        let batch_states = create_test_batch_states();
        let batch_seq = create_test_batch_sequence();
        let manager = SessionManager::new(batch_states, batch_seq, SessionMode::Reusable);

        let handle1 = manager.acquire_session("session-1").await.unwrap();
        let handle2 = manager.acquire_session("session-2").await.unwrap();
        let handle3 = manager.acquire_session("session-3").await.unwrap();
        let handle4 = manager.acquire_session("session-4").await.unwrap();

        assert_eq!(manager.session_count().await, 4);

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        manager.acquire_session("session-2").await.unwrap();

        let new_handle = manager.acquire_session("session-5").await.unwrap();

        assert_eq!(manager.session_count().await, 4);

        assert_eq!(new_handle.slot_index, handle1.slot_index);
    }
}
