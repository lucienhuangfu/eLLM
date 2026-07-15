use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use tokio::sync::Mutex as TokioMutex;

use crate::num_traits::FromNumber;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch::BatchSequence;

use super::slot::SlotState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionMode {
    Reusable,
    NonReusable,
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

struct SessionMeta {
    token_count: usize,
}

struct LruList {
    slots: Vec<usize>,
}

impl LruList {
    fn new(n: usize) -> Self {
        Self {
            slots: (0..n).collect(),
        }
    }

    fn touch(&mut self, idx: usize) {
        if let Some(pos) = self.slots.iter().position(|&x| x == idx) {
            self.slots.remove(pos);
        }
        self.slots.insert(0, idx);
    }

    fn evict_tail(&mut self) -> usize {
        self.slots.pop().expect("LRU list is empty")
    }

    fn insert_head(&mut self, idx: usize) {
        if let Some(pos) = self.slots.iter().position(|&x| x == idx) {
            self.slots.remove(pos);
        }
        self.slots.insert(0, idx);
    }

    fn detach(&mut self, idx: usize) {
        if let Some(pos) = self.slots.iter().position(|&x| x == idx) {
            self.slots.remove(pos);
        }
    }
}

pub struct SlotManager<T: Copy + FromNumber> {
    batch_states: Arc<SharedMut<Vec<SlotState>>>,
    batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    lru: StdMutex<LruList>,
    session_map: TokioMutex<HashMap<String, usize>>,
    reserved_slots: Arc<TokioMutex<HashMap<String, (usize, Arc<AtomicBool>)>>>,
    meta: StdMutex<Vec<SessionMeta>>,
    mode: SessionMode,
    reuse_timeout: Duration,
}

impl<T: Copy + FromNumber> SlotManager<T> {
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
            lru: StdMutex::new(LruList::new(num_slots)),
            session_map: TokioMutex::new(HashMap::new()),
            reserved_slots: Arc::new(TokioMutex::new(HashMap::new())),
            meta: StdMutex::new(meta),
            mode,
            reuse_timeout: Duration::from_millis(reuse_timeout_ms),
        }
    }

    pub fn batch_list(&self) -> Arc<SharedMut<Vec<SlotState>>> {
        Arc::clone(&self.batch_states)
    }

    pub fn with_slots<R>(&self, f: impl FnOnce(&[SlotState]) -> R) -> R {
        self.batch_states.with(|v| f(v.as_slice()))
    }

    pub fn with_slots_mut<R>(&self, f: impl FnOnce(&mut [SlotState]) -> R) -> R {
        self.batch_states.with_mut(|v| f(v.as_mut_slice()))
    }

    pub fn detach_from_lru(&self, idx: usize) {
        self.lru.lock().unwrap().detach(idx);
    }

    pub async fn acquire_session(&self, session_id: &str) -> Result<SessionHandle, super::slot::SlotError> {
        let mut map = self.session_map.lock().await;

        if let Some(&slot_index) = map.get(session_id) {
            return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
        }

        let reserved = {
            let mut r = self.reserved_slots.lock().await;
            r.remove(session_id)
        };

        if let Some((slot_index, cancel_flag)) = reserved {
            cancel_flag.store(true, Ordering::Release);
            map.insert(session_id.to_string(), slot_index);
            return Ok(SessionHandle::reused(session_id.to_string(), slot_index));
        }

        let slot_index = self.lru.lock().unwrap().evict_tail();

        if let Some(old_id) = map
            .iter()
            .find(|(_, &idx)| idx == slot_index)
            .map(|(k, _)| k.clone())
        {
            map.remove(&old_id);
        }
        map.insert(session_id.to_string(), slot_index);

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
        let lru_ptr = &self.lru as *const StdMutex<LruList> as usize;
        let timeout = self.reuse_timeout;

        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            if cancel_flag.load(Ordering::Acquire) {
                return;
            }

            let mut reserved = reserved_slots.lock().await;
            if let Some((idx, _)) = reserved.remove(&session_id_owned) {
                batch_states.with_mut(|slots| {
                    slots[idx].reset_to_start();
                });
                let lru = unsafe { &*(lru_ptr as *const StdMutex<LruList>) };
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
    use crate::runtime::batch::BatchSequence;
    use crate::runtime::session::slot::Phase;

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

    fn create_test_manager_with_buffer(
        batch_size: usize,
        timeout_ms: u64,
        buffer: &mut Vec<usize>,
    ) -> Arc<SlotManager<f16>> {
        let dir = model_dir();
        let seq_len = 1024;
        buffer.resize(batch_size * seq_len, 0);
        let batch_sequences = Arc::new(SharedMut::new(
            BatchSequence::<f16>::new(
                buffer.as_mut_ptr(),
                batch_size,
                seq_len,
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
        assert_eq!(phase, Phase::Start);
    }

    #[tokio::test]
    async fn test_calculate_delta_full_prefix_match() {
        let mut buffer = Vec::new();
        let manager = create_test_manager_with_buffer(4, 5000, &mut buffer);

        let session_id = "prefix_test";
        let handle = manager.acquire_session(session_id).await.unwrap();
        let slot_idx = handle.slot_index;

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        manager.batch_sequences.with_mut(|bs| {
            bs.write_tokens(slot_idx, &tokens, 1.0).unwrap();
        });

        manager.release_session(session_id, tokens.len()).await;

        let handle2 = manager.acquire_session(session_id).await.unwrap();
        assert!(handle2.is_reused);

        let new_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 11, 12, 13];
        let delta = manager.calculate_delta(session_id, &new_tokens).await;
        assert!(delta.is_some());
        let (prefix_len, delta_tokens) = delta.unwrap();
        assert_eq!(prefix_len, 5);
        assert_eq!(delta_tokens, vec![11, 12, 13]);
    }

    #[tokio::test]
    async fn test_calculate_delta_no_prefix_match() {
        let mut buffer = Vec::new();
        let manager = create_test_manager_with_buffer(4, 5000, &mut buffer);

        let session_id = "no_prefix_test";
        let handle = manager.acquire_session(session_id).await.unwrap();
        let slot_idx = handle.slot_index;

        let tokens: Vec<u32> = vec![100, 200, 300];
        manager.batch_sequences.with_mut(|bs| {
            bs.write_tokens(slot_idx, &tokens, 1.0).unwrap();
        });

        manager.release_session(session_id, tokens.len()).await;
        manager.acquire_session(session_id).await.unwrap();

        let new_tokens: Vec<u32> = vec![999, 888];
        let delta = manager.calculate_delta(session_id, &new_tokens).await;
        assert!(delta.is_none());
    }

    #[tokio::test]
    async fn test_calculate_delta_new_tokens_shorter() {
        let mut buffer = Vec::new();
        let manager = create_test_manager_with_buffer(4, 5000, &mut buffer);

        let session_id = "shorter_test";
        let handle = manager.acquire_session(session_id).await.unwrap();
        let slot_idx = handle.slot_index;

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        manager.batch_sequences.with_mut(|bs| {
            bs.write_tokens(slot_idx, &tokens, 1.0).unwrap();
        });

        manager.release_session(session_id, tokens.len()).await;
        manager.acquire_session(session_id).await.unwrap();

        let new_tokens: Vec<u32> = vec![1, 2, 3];
        let delta = manager.calculate_delta(session_id, &new_tokens).await;
        assert!(delta.is_some());
        let (prefix_len, delta_tokens) = delta.unwrap();
        assert_eq!(prefix_len, 3);
        assert!(delta_tokens.is_empty());
    }

    #[tokio::test]
    async fn test_calculate_delta_exact_match() {
        let mut buffer = Vec::new();
        let manager = create_test_manager_with_buffer(4, 5000, &mut buffer);

        let session_id = "exact_test";
        let handle = manager.acquire_session(session_id).await.unwrap();
        let slot_idx = handle.slot_index;

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        manager.batch_sequences.with_mut(|bs| {
            bs.write_tokens(slot_idx, &tokens, 1.0).unwrap();
        });

        manager.release_session(session_id, tokens.len()).await;
        manager.acquire_session(session_id).await.unwrap();

        let delta = manager.calculate_delta(session_id, &tokens).await;
        assert!(delta.is_some());
        let (prefix_len, delta_tokens) = delta.unwrap();
        assert_eq!(prefix_len, 5);
        assert!(delta_tokens.is_empty());
    }

    #[tokio::test]
    async fn test_calculate_delta_zero_token_count() {
        let manager = create_test_manager(4, 1000);

        let h = manager.acquire_session("zero_token").await.unwrap();
        manager.release_session("zero_token", 0).await;

        let h2 = manager.acquire_session("zero_token").await.unwrap();
        assert!(h2.is_reused);

        let delta = manager.calculate_delta("zero_token", &[1, 2, 3]).await;
        assert!(delta.is_none());
    }

    #[tokio::test]
    async fn test_calculate_delta_session_not_found() {
        let manager = create_test_manager(4, 1000);
        let delta = manager.calculate_delta("nonexistent", &[1, 2, 3]).await;
        assert!(delta.is_none());
    }
}
