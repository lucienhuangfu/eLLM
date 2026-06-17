use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::oneshot;
use tokio::sync::Mutex;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::{SlotError, SlotResult};
use crate::runtime::state::types::SequenceState;

pub struct SlotAllocator {
    free_slots: Arc<Mutex<VecDeque<usize>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    slot_timers: Arc<Mutex<HashMap<usize, oneshot::Sender<()>>>>,
    timeout_duration: Duration,
}

impl SlotAllocator {
    pub fn new(
        batch_states: Arc<SharedMut<Vec<SequenceState>>>,
        timeout_duration: Duration,
    ) -> Self {
        let initial_free_slots: VecDeque<usize> = batch_states.with(|batch_states_ref| {
            batch_states_ref
                .iter()
                .enumerate()
                .filter_map(|(i, record)| record.is_available().then_some(i))
                .collect()
        });

        Self {
            free_slots: Arc::new(Mutex::new(initial_free_slots)),
            batch_states,
            slot_timers: Arc::new(Mutex::new(HashMap::new())),
            timeout_duration,
        }
    }

    pub async fn allocate(&self) -> SlotResult<usize> {
        let mut free_slots = self.free_slots.lock().await;
        let slot_index = free_slots.pop_front().ok_or(SlotError::SlotQueueEmpty)?;
        drop(free_slots);

        self.cancel_timer(slot_index).await;

        Ok(slot_index)
    }

    pub async fn allocate_preferred(&self, preferred_slot: usize) -> SlotResult<usize> {
        {
            let mut free_slots = self.free_slots.lock().await;
            if let Some(pos) = free_slots.iter().position(|&s| s == preferred_slot) {
                free_slots.remove(pos);
                drop(free_slots);
                self.cancel_timer(preferred_slot).await;
                return Ok(preferred_slot);
            }
        }

        if self.cancel_timer(preferred_slot).await {
            return Ok(preferred_slot);
        }

        Err(SlotError::SlotQueueEmpty)
    }

    pub async fn release(&self, slot_index: usize) {
        let (tx, rx) = oneshot::channel();
        {
            let mut timers = self.slot_timers.lock().await;
            timers.insert(slot_index, tx);
        }

        let batch_states = self.batch_states.clone();
        let free_slots = self.free_slots.clone();
        let slot_timers = self.slot_timers.clone();
        let timeout_duration = self.timeout_duration;

        tokio::spawn(async move {
            tokio::select! {
                _ = rx => {
                }
                _ = tokio::time::sleep(timeout_duration) => {
                    batch_states.with_mut(|batch_list| {
                        if let Some(record) = batch_list.get_mut(slot_index) {
                            record.reset_to_start();
                        }
                    });
                    free_slots.lock().await.push_back(slot_index);
                }
            }

            let mut timers = slot_timers.lock().await;
            timers.remove(&slot_index);
        });
    }

    pub async fn cancel_timer(&self, slot_index: usize) -> bool {
        if let Some(tx) = self.slot_timers.lock().await.remove(&slot_index) {
            let _ = tx.send(());
            true
        } else {
            false
        }
    }

    pub async fn available_count(&self) -> usize {
        let free_slots = self.free_slots.lock().await;
        free_slots.len()
    }

    pub async fn is_timed(&self, slot_index: usize) -> bool {
        let timers = self.slot_timers.lock().await;
        timers.contains_key(&slot_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_batch_states() -> Arc<SharedMut<Vec<SequenceState>>> {
        Arc::new(SharedMut::new(vec![SequenceState::new_start_state(); 4]))
    }

    #[tokio::test]
    async fn test_allocate_and_release() {
        let batch_states = create_test_batch_states();
        let allocator = SlotAllocator::new(batch_states, Duration::from_millis(100));

        assert_eq!(allocator.available_count().await, 4);

        let slot1 = allocator.allocate().await.unwrap();
        assert_eq!(allocator.available_count().await, 3);

        allocator.release(slot1).await;
        assert_eq!(allocator.available_count().await, 3);
        assert!(allocator.is_timed(slot1).await);

        tokio::time::sleep(Duration::from_millis(150)).await;

        assert_eq!(allocator.available_count().await, 4);
        assert!(!allocator.is_timed(slot1).await);
    }

    #[tokio::test]
    async fn test_allocate_preferred_timed() {
        let batch_states = create_test_batch_states();
        let allocator = SlotAllocator::new(batch_states, Duration::from_millis(100));

        let slot = allocator.allocate().await.unwrap();

        allocator.release(slot).await;
        assert!(allocator.is_timed(slot).await);

        let reused_slot = allocator.allocate_preferred(slot).await.unwrap();
        assert_eq!(reused_slot, slot);
        assert!(!allocator.is_timed(slot).await);
    }

    #[tokio::test]
    async fn test_allocate_preferred_free() {
        let batch_states = create_test_batch_states();
        let allocator = SlotAllocator::new(batch_states, Duration::from_millis(100));

        let slot = allocator.allocate().await.unwrap();

        allocator.release(slot).await;
        tokio::time::sleep(Duration::from_millis(150)).await;

        assert_eq!(allocator.available_count().await, 4);

        let reused_slot = allocator.allocate_preferred(slot).await.unwrap();
        assert_eq!(reused_slot, slot);
        assert_eq!(allocator.available_count().await, 3);
    }

    #[tokio::test]
    async fn test_allocate_preferred_in_use() {
        let batch_states = create_test_batch_states();
        let allocator = SlotAllocator::new(batch_states, Duration::from_millis(100));

        let slot1 = allocator.allocate().await.unwrap();
        let slot2 = allocator.allocate().await.unwrap();

        let result = allocator.allocate_preferred(slot1).await;
        assert!(result.is_err());

        allocator.release(slot1).await;

        let reused = allocator.allocate_preferred(slot1).await.unwrap();
        assert_eq!(reused, slot1);
    }
}
