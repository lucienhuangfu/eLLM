use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::SlotResult;
use crate::runtime::session::allocator::SlotAllocator;
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::types::SequenceState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionMode {
    Reusable,
    NonReusable,
}

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
    pub fn can_reuse(&self) -> bool {
        self.mode == SessionMode::Reusable && self.slot_index.is_some() && !self.is_active
    }

    pub fn activate(&mut self) {
        self.is_active = true;
        self.last_accessed = Instant::now();
    }

    pub fn deactivate(&mut self) {
        self.is_active = false;
        self.last_accessed = Instant::now();
    }
}

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
        timeout_duration: Duration,
    ) -> Self {
        let slot_allocator = Arc::new(SlotAllocator::new(batch_states, timeout_duration));

        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            slot_allocator,
            batch_sequences,
            max_sessions,
        }
    }

    pub async fn acquire_session(
        &self,
        session_id: &str,
        mode: SessionMode,
    ) -> SlotResult<SessionHandle> {
        let mut sessions = self.sessions.lock().await;

        if let Some(session) = sessions.get_mut(session_id) {
            if let Some(preferred_slot) = session.slot_index {
                match self.slot_allocator.allocate_preferred(preferred_slot).await {
                    Ok(_) => {
                        session.activate();
                        return Ok(SessionHandle::reused(
                            session_id.to_string(),
                            preferred_slot,
                        ));
                    }
                    Err(_) => {}
                }
            }
        }

        if sessions.len() >= self.max_sessions {
            self.evict_lru_session(&mut sessions).await;
        }

        let slot_index = self.slot_allocator.allocate().await?;

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

    pub async fn release_session(&self, session_id: &str, token_count: usize) {
        let mut sessions = self.sessions.lock().await;

        if let Some(session) = sessions.get_mut(session_id) {
            session.deactivate();
            session.token_count = token_count;

            if session.mode == SessionMode::NonReusable {
                let slot_index = session.slot_index.take();
                if let Some(idx) = slot_index {
                    self.slot_allocator.release(idx).await;
                }
                sessions.remove(session_id);
            } else {
                if let Some(idx) = session.slot_index {
                    self.slot_allocator.release(idx).await;
                }
            }
        }
    }

    pub async fn get_cached_tokens(&self, session_id: &str) -> Option<(usize, usize)> {
        let sessions = self.sessions.lock().await;
        sessions
            .get(session_id)
            .filter(|s| s.token_count > 0)
            .and_then(|s| s.slot_index.map(|idx| (idx, s.token_count)))
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
        let manager = SessionManager::new(batch_states, batch_seq, 10, Duration::from_millis(100));

        let handle1 = manager
            .acquire_session("test-session", SessionMode::Reusable)
            .await
            .unwrap();
        assert_eq!(handle1.session_id, "test-session");
        assert!(!handle1.is_reused);

        manager.release_session("test-session", 10).await;

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
        let manager = SessionManager::new(batch_states, batch_seq, 10, Duration::from_millis(100));

        let handle1 = manager
            .acquire_session("test-session", SessionMode::NonReusable)
            .await
            .unwrap();

        manager.release_session("test-session", 10).await;

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
        let manager = SessionManager::new(batch_states, batch_seq, 10, Duration::from_millis(100));

        let handle = manager
            .acquire_session("active-session", SessionMode::Reusable)
            .await
            .unwrap();

        let handle2 = manager
            .acquire_session("active-session", SessionMode::Reusable)
            .await
            .unwrap();
        assert_ne!(handle2.slot_index, handle.slot_index);
    }

    #[tokio::test]
    async fn test_reuse_after_timeout() {
        let batch_states = create_test_batch_states();
        let batch_seq = create_test_batch_sequence();
        let manager = SessionManager::new(batch_states, batch_seq, 10, Duration::from_millis(50));

        let handle1 = manager
            .acquire_session("test-session", SessionMode::Reusable)
            .await
            .unwrap();
        let slot1 = handle1.slot_index;

        manager.release_session("test-session", 10).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        let handle2 = manager
            .acquire_session("test-session", SessionMode::Reusable)
            .await
            .unwrap();
        assert_eq!(handle2.slot_index, slot1);
        assert!(handle2.is_reused);
    }
}
