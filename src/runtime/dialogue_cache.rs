use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch_sequence::BatchSequence;
use crate::runtime::cache_strategy::{DialogueEntry, LruCacheStrategy};
use crate::runtime::slot_manager::SlotManager;

pub struct DialogueCache {
    strategy: Arc<LruCacheStrategy>,
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
}

impl DialogueCache {
    pub fn new(
        slot_manager: Arc<SlotManager>,
        batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
        retention_duration: Duration,
        max_entries: usize,
    ) -> Self {
        let strategy = Arc::new(LruCacheStrategy::new(
            slot_manager,
            retention_duration,
            max_entries,
        ));

        Self {
            strategy,
            batch_sequences,
        }
    }

    pub async fn get(&self, dialogue_id: &str) -> Option<DialogueEntry> {
        self.strategy.get(dialogue_id).await
    }

    pub async fn insert(&self, dialogue_id: String, slot_index: usize, token_count: usize) {
        self.strategy
            .insert(dialogue_id, slot_index, token_count)
            .await
    }

    pub async fn insert_batch(&self, entries: Vec<(String, usize, usize)>) {
        self.strategy.insert_batch(entries).await
    }

    pub async fn remove(&self, dialogue_id: &str) {
        self.strategy.remove(dialogue_id).await
    }

    pub async fn remove_batch(&self, dialogue_ids: &[&str]) {
        self.strategy.remove_batch(dialogue_ids).await
    }

    pub async fn cleanup(&self, now: Instant) {
        self.strategy.cleanup(now).await
    }

    pub async fn calculate_delta(
        &self,
        entry: &DialogueEntry,
        new_tokens: &[u32],
    ) -> (usize, Vec<u32>) {
        let cached_tokens = self
            .batch_sequences
            .with(|batch_seq| batch_seq.token_ids(entry.slot_index, 0, entry.token_count));

        let min_len = cached_tokens.len().min(new_tokens.len());
        let mut prefix_len = 0;

        while prefix_len < min_len && cached_tokens[prefix_len] == new_tokens[prefix_len] {
            prefix_len += 1;
        }

        (prefix_len, new_tokens[prefix_len..].to_vec())
    }

    pub async fn find_common_prefix(
        &self,
        dialogue_id: &str,
        new_tokens: &[u32],
    ) -> Option<(DialogueEntry, usize, Vec<u32>)> {
        let entry = self.get(dialogue_id).await?;
        let (prefix_len, delta_tokens) = self.calculate_delta(&entry, new_tokens).await;

        if prefix_len > 0 {
            Some((entry, prefix_len, delta_tokens))
        } else {
            None
        }
    }

    pub async fn entry_count(&self) -> usize {
        self.strategy.entry_count().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::scheduling::SequenceState;
    use std::time::Duration;

    fn create_test_slot_manager() -> Arc<SlotManager> {
        let batch_states = Arc::new(SharedMut::new(vec![SequenceState::new_start_state(); 4]));
        Arc::new(SlotManager::new(batch_states))
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let slot_manager = create_test_slot_manager();
        let batch_seq = Arc::new(SharedMut::new(unsafe {
            BatchSequence {
                sequences: std::ptr::null_mut(),
                batch_temperature: vec![1.0; 4],
                row_size: 4,
                col_size: 64,
                tokenizer: unimplemented!(),
                chat_template: unimplemented!(),
            }
        }));
        let cache = DialogueCache::new(slot_manager, batch_seq, Duration::from_secs(10), 4);

        cache.insert("dialogue-1".to_string(), 0, 10).await;

        let entry = cache.get("dialogue-1").await;
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().token_count, 10);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let slot_manager = create_test_slot_manager();
        let batch_seq = Arc::new(SharedMut::new(unsafe {
            BatchSequence {
                sequences: std::ptr::null_mut(),
                batch_temperature: vec![1.0; 4],
                row_size: 4,
                col_size: 64,
                tokenizer: unimplemented!(),
                chat_template: unimplemented!(),
            }
        }));
        let cache = DialogueCache::new(slot_manager, batch_seq, Duration::from_millis(10), 2);

        cache.insert("dialogue-1".to_string(), 0, 10).await;
        cache.insert("dialogue-2".to_string(), 1, 20).await;

        tokio::time::sleep(Duration::from_millis(20)).await;

        cache.cleanup(Instant::now()).await;

        cache.insert("dialogue-3".to_string(), 2, 30).await;

        assert_eq!(cache.entry_count().await, 2);
    }

    #[tokio::test]
    async fn test_calculate_delta() {
        let slot_manager = create_test_slot_manager();
        let mut storage = vec![1usize, 2, 3, 4, 5, 0, 0, 0];
        let batch_seq = Arc::new(SharedMut::new(unsafe {
            BatchSequence {
                sequences: storage.as_mut_ptr(),
                batch_temperature: vec![1.0; 4],
                row_size: 4,
                col_size: 2,
                tokenizer: unimplemented!(),
                chat_template: unimplemented!(),
            }
        }));
        let cache = DialogueCache::new(slot_manager, batch_seq, Duration::from_secs(10), 4);

        let entry = DialogueEntry {
            dialogue_id: "test".to_string(),
            slot_index: 0,
            token_count: 2,
            last_accessed_at: Instant::now(),
            in_lru: false,
            lru_index: None,
        };

        let new_tokens = &[1u32, 2, 6, 7];
        let (prefix_len, delta) = cache.calculate_delta(&entry, new_tokens).await;

        assert_eq!(prefix_len, 2);
        assert_eq!(delta, vec![6, 7]);
    }

    #[tokio::test]
    async fn test_batch_insert() {
        let slot_manager = create_test_slot_manager();
        let batch_seq = Arc::new(SharedMut::new(unsafe {
            BatchSequence {
                sequences: std::ptr::null_mut(),
                batch_temperature: vec![1.0; 4],
                row_size: 4,
                col_size: 64,
                tokenizer: unimplemented!(),
                chat_template: unimplemented!(),
            }
        }));
        let cache = DialogueCache::new(slot_manager, batch_seq, Duration::from_secs(10), 10);

        let entries = vec![
            ("dialogue-1".to_string(), 0, 10),
            ("dialogue-2".to_string(), 1, 20),
            ("dialogue-3".to_string(), 2, 30),
        ];
        cache.insert_batch(entries).await;

        assert_eq!(cache.entry_count().await, 3);
    }
}
