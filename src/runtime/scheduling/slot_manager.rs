use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::{Mutex, Semaphore};

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::{SlotError, SlotResult};
use crate::runtime::scheduling::SequenceState;

pub struct SlotManager {
    free_slots: Arc<Mutex<VecDeque<usize>>>,
    available_slots: Arc<Semaphore>,
    slot_to_dialogue: Arc<Mutex<HashMap<usize, String>>>,
    dialogue_to_slot: Arc<Mutex<HashMap<String, usize>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
}

impl SlotManager {
    pub fn new(batch_states: Arc<SharedMut<Vec<SequenceState>>>) -> Self {
        let initial_free_slots: VecDeque<usize> = batch_states.with(|batch_states_ref| {
            batch_states_ref
                .iter()
                .enumerate()
                .filter_map(|(i, record)| record.is_available().then_some(i))
                .collect()
        });
        let initial_permits = initial_free_slots.len();

        Self {
            free_slots: Arc::new(Mutex::new(initial_free_slots)),
            available_slots: Arc::new(Semaphore::new(initial_permits)),
            slot_to_dialogue: Arc::new(Mutex::new(HashMap::new())),
            dialogue_to_slot: Arc::new(Mutex::new(HashMap::new())),
            batch_states,
        }
    }

    pub async fn acquire_slot(&self, dialogue_id: Option<&str>) -> SlotResult<usize> {
        let permit = self
            .available_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| SlotError::AllocatorUnavailable)?;

        let slot_index = {
            let mut free_slots = self.free_slots.lock().await;
            free_slots.pop_front().ok_or(SlotError::SlotQueueEmpty)?
        };

        if let Some(id) = dialogue_id {
            let mut slot_to_dialogue = self.slot_to_dialogue.lock().await;
            let mut dialogue_to_slot = self.dialogue_to_slot.lock().await;
            slot_to_dialogue.insert(slot_index, id.to_string());
            dialogue_to_slot.insert(id.to_string(), slot_index);
        }

        permit.forget();
        Ok(slot_index)
    }

    pub async fn release_slot(&self, slot_index: usize, release_permit: bool) {
        self.batch_states.with_mut(|batch_list| {
            if let Some(record) = batch_list.get_mut(slot_index) {
                record.reset_to_start();
            }
        });

        let mut slot_to_dialogue = self.slot_to_dialogue.lock().await;
        let mut dialogue_to_slot = self.dialogue_to_slot.lock().await;
        
        if let Some(dialogue_id) = slot_to_dialogue.remove(&slot_index) {
            dialogue_to_slot.remove(&dialogue_id);
        }

        let mut free_slots = self.free_slots.lock().await;
        free_slots.push_back(slot_index);
        drop(free_slots);

        if release_permit {
            self.available_slots.add_permits(1);
        }
    }

    pub async fn release_by_dialogue(&self, dialogue_id: &str) -> Option<usize> {
        let mut slot_to_dialogue = self.slot_to_dialogue.lock().await;
        let mut dialogue_to_slot = self.dialogue_to_slot.lock().await;

        let slot_index = dialogue_to_slot.remove(dialogue_id)?;
        slot_to_dialogue.remove(&slot_index);

        self.batch_states.with_mut(|batch_list| {
            if let Some(record) = batch_list.get_mut(slot_index) {
                record.reset_to_start();
            }
        });

        let mut free_slots = self.free_slots.lock().await;
        free_slots.push_back(slot_index);
        drop(free_slots);

        self.available_slots.add_permits(1);

        Some(slot_index)
    }

    pub async fn get_dialogue_for_slot(&self, slot_index: usize) -> Option<String> {
        let slot_to_dialogue = self.slot_to_dialogue.lock().await;
        slot_to_dialogue.get(&slot_index).cloned()
    }

    pub async fn get_slot_for_dialogue(&self, dialogue_id: &str) -> Option<usize> {
        let dialogue_to_slot = self.dialogue_to_slot.lock().await;
        dialogue_to_slot.get(dialogue_id).copied()
    }

    pub async fn is_slot_owned_by(&self, slot_index: usize, dialogue_id: &str) -> bool {
        let slot_to_dialogue = self.slot_to_dialogue.lock().await;
        slot_to_dialogue.get(&slot_index) == Some(&dialogue_id.to_string())
    }

    pub fn available_permits(&self) -> usize {
        self.available_slots.available_permits()
    }
}