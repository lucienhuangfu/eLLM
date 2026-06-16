use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::SlotResult;
use crate::runtime::session::allocator::SlotAllocator;
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::types::SequenceState;

/// 会话模式枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionMode {
    /// 复用模式：相同 session_id 复用槽位，保留映射
    Reusable,
    /// 不复用模式：每次请求分配新槽位，清除映射
    NonReusable,
}

/// 对话会话结构
#[derive(Debug, Clone)]
pub struct DialogueSession {
    pub session_id: String,
    pub mode: SessionMode,
    pub slot_index: Option<usize>,
    pub token_count: usize,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub is_active: bool,
}

impl DialogueSession {
    /// 检查是否可以复用此会话的槽位
    pub fn can_reuse(&self) -> bool {
        self.mode == SessionMode::Reusable && self.slot_index.is_some() && !self.is_active
    }

    /// 标记会话为活跃状态
    pub fn activate(&mut self) {
        self.is_active = true;
        self.last_accessed = Instant::now();
    }

    /// 标记会话为非活跃状态
    pub fn deactivate(&mut self) {
        self.is_active = false;
        self.last_accessed = Instant::now();
    }
}

/// 会话句柄
#[derive(Debug, Clone)]
pub struct SessionHandle {
    pub session_id: String,
    pub slot_index: usize,
    pub is_reused: bool,
}

impl SessionHandle {
    pub fn new(session_id: String, slot_index: usize) -> Self {
        Self {
            session_id,
            slot_index,
            is_reused: false,
        }
    }

    pub fn reused(session_id: String, slot_index: usize) -> Self {
        Self {
            session_id,
            slot_index,
            is_reused: true,
        }
    }
}

/// 会话管理器 - 统一管理会话生命周期、槽位绑定和 token 缓存
pub struct SessionManager<T> {
    sessions: Arc<Mutex<HashMap<String, DialogueSession>>>,
    slot_allocator: Arc<SlotAllocator>,
    batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    max_sessions: usize,
}

impl<T> SessionManager<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    pub fn new(
        batch_states: Arc<SharedMut<Vec<SequenceState>>>,
        batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
        max_sessions: usize,
    ) -> Self {
        let slot_allocator = Arc::new(SlotAllocator::new(batch_states));

        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            slot_allocator,
            batch_sequences,
            max_sessions,
        }
    }

    /// 获取或创建会话
    pub async fn acquire_session(
        &self,
        session_id: &str,
        mode: SessionMode,
    ) -> SlotResult<SessionHandle> {
        let mut sessions = self.sessions.lock().await;

        // 尝试查找现有会话
        if let Some(session) = sessions.get_mut(session_id) {
            if session.can_reuse() {
                // 复用模式：重用现有槽位
                let slot_index = session.slot_index.unwrap();
                session.activate();
                return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
            }
        }

        // 检查会话数量限制
        if sessions.len() >= self.max_sessions {
            // 简单的 LRU 清理：移除最久未访问的非活跃会话
            self.evict_lru_session(&mut sessions).await;
        }

        // 分配新槽位
        let slot_index = self.slot_allocator.allocate().await?;

        // 创建新会话
        let new_session = DialogueSession {
            session_id: session_id.to_string(),
            mode,
            slot_index: Some(slot_index),
            token_count: 0,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            is_active: true,
        };

        sessions.insert(session_id.to_string(), new_session);
        Ok(SessionHandle::new(session_id.to_string(), slot_index))
    }

    /// 释放会话
    pub async fn release_session(&self, session_id: &str, token_count: usize) {
        let mut sessions = self.sessions.lock().await;

        if let Some(session) = sessions.get_mut(session_id) {
            session.deactivate();
            session.token_count = token_count;

            // 不复用模式：立即清理
            if session.mode == SessionMode::NonReusable {
                let slot_index = session.slot_index.take();
                if let Some(idx) = slot_index {
                    self.slot_allocator.release(idx).await;
                }
                sessions.remove(session_id);
            }
            // 复用模式：保留会话和槽位映射
        }
    }

    /// 获取会话的 token 缓存信息（用于增量预填充）
    pub async fn get_cached_tokens(&self, session_id: &str) -> Option<(usize, usize)> {
        let sessions = self.sessions.lock().await;
        sessions
            .get(session_id)
            .filter(|s| s.token_count > 0)
            .and_then(|s| s.slot_index.map(|idx| (idx, s.token_count)))
    }

    /// 计算前缀匹配并返回 delta tokens
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

    /// LRU 清理：移除最久未访问的非活跃会话
    async fn evict_lru_session(&self, sessions: &mut HashMap<String, DialogueSession>) {
        let mut oldest_session: Option<(String, Instant)> = None;

        for (id, session) in sessions.iter() {
            if !session.is_active {
                if let Some((_, oldest_time)) = &oldest_session {
                    if session.last_accessed < *oldest_time {
                        oldest_session = Some((id.clone(), session.last_accessed));
                    }
                } else {
                    oldest_session = Some((id.clone(), session.last_accessed));
                }
            }
        }

        if let Some((oldest_id, _)) = oldest_session {
            if let Some(session) = sessions.remove(&oldest_id) {
                if let Some(idx) = session.slot_index {
                    self.slot_allocator.release(idx).await;
                }
            }
        }
    }

    /// 获取当前会话数量
    pub async fn session_count(&self) -> usize {
        let sessions = self.sessions.lock().await;
        sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::state::types::Phase;

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
        let manager = SessionManager::new(batch_states, batch_seq, 10);

        // 第一次获取会话
        let handle1 = manager
            .acquire_session("test-session", SessionMode::Reusable)
            .await
            .unwrap();
        assert_eq!(handle1.session_id, "test-session");
        assert!(!handle1.is_reused);

        // 释放会话
        manager.release_session("test-session", 10).await;

        // 第二次获取相同会话，应该复用
        let handle2 = manager
            .acquire_session("test-session", SessionMode::Reusable)
            .await
            .unwrap();
        assert_eq!(handle2.slot_index, handle1.slot_index);
        assert!(handle2.is_reused);
    }

    #[tokio::test]
    async fn test_non_reusable_mode() {
        let batch_states = create_test_batch_states();
        let batch_seq = create_test_batch_sequence();
        let manager = SessionManager::new(batch_states, batch_seq, 10);

        // 第一次获取会话
        let handle1 = manager
            .acquire_session("test-session", SessionMode::NonReusable)
            .await
            .unwrap();

        // 释放会话
        manager.release_session("test-session", 10).await;

        // 第二次获取相同会话，应该分配新槽位
        let handle2 = manager
            .acquire_session("test-session", SessionMode::NonReusable)
            .await
            .unwrap();
        assert_ne!(handle2.slot_index, handle1.slot_index);
        assert!(!handle2.is_reused);
    }

    #[tokio::test]
    async fn test_concurrent_access_prevention() {
        let batch_states = create_test_batch_states();
        let batch_seq = create_test_batch_sequence();
        let manager = SessionManager::new(batch_states, batch_seq, 10);

        // 获取会话但不释放
        let handle = manager
            .acquire_session("active-session", SessionMode::Reusable)
            .await
            .unwrap();

        // 再次尝试获取相同会话，应该分配新槽位（因为原会话仍活跃）
        let handle2 = manager
            .acquire_session("active-session", SessionMode::Reusable)
            .await
            .unwrap();
        assert_ne!(handle2.slot_index, handle.slot_index);
    }
}
