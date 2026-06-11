use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use tokio::sync::{Mutex, Semaphore};

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::scheduling::SequenceState;

pub struct SlotManager {
    free_slots: Arc<Mutex<VecDeque<usize>>>,
    available_slots: Arc<Semaphore>,
    slot_owners: Arc<Mutex<HashMap<usize, String>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
}

impl SlotManager {
    pub fn new(batch_states: Arc<SharedMut<Vec<SequenceState>>>) -> Self {
        let initial_free_slots: VecDeque<usize> = batch_states.with(|batch_states_ref| {
            batch_states_ref
                .iter()
                .enumerate()
                .filter_map(|(i, record)| (record.is_available()).then_some(i))
                .collect()
        });
        let initial_permits = initial_free_slots.len();

        Self {
            free_slots: Arc::new(Mutex::new(initial_free_slots)),
            available_slots: Arc::new(Semaphore::new(initial_permits)),
            slot_owners: Arc::new(Mutex::new(HashMap::new())),
            batch_states,
        }
    }

    pub async fn acquire_slot(
        &self,
        dialogue_id: Option<&str>,
    ) -> Result<usize, axum::response::Response> {
        let permit = self
            .available_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Slot allocator unavailable".to_string(),
                )
                    .into_response()
            })?;

        let slot_index = {
            let mut free_slots = self.free_slots.lock().await;
            free_slots.pop_front().ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Slot queue empty while permit acquired".to_string(),
                )
                    .into_response()
            })?
        };

        if let Some(id) = dialogue_id {
            let mut slot_owners = self.slot_owners.lock().await;
            slot_owners.insert(slot_index, id.to_string());
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

        let mut slot_owners = self.slot_owners.lock().await;
        slot_owners.remove(&slot_index);

        let mut free_slots = self.free_slots.lock().await;
        free_slots.push_back(slot_index);
        drop(free_slots);

        if release_permit {
            self.available_slots.add_permits(1);
        }
    }

    pub async fn release_by_dialogue(&self, dialogue_id: &str) -> Option<usize> {
        let mut slot_owners = self.slot_owners.lock().await;
        let mut found_slot: Option<usize> = None;

        for (&slot_idx, id) in slot_owners.iter() {
            if id == dialogue_id {
                found_slot = Some(slot_idx);
                break;
            }
        }

        if let Some(slot_index) = found_slot {
            slot_owners.remove(&slot_index);

            self.batch_states.with_mut(|batch_list| {
                if let Some(record) = batch_list.get_mut(slot_index) {
                    record.reset_to_start();
                }
            });

            let mut free_slots = self.free_slots.lock().await;
            free_slots.push_back(slot_index);
            drop(free_slots);

            self.available_slots.add_permits(1);
        }

        found_slot
    }

    pub async fn get_dialogue_for_slot(&self, slot_index: usize) -> Option<String> {
        let slot_owners = self.slot_owners.lock().await;
        slot_owners.get(&slot_index).cloned()
    }

    pub async fn get_slot_for_dialogue(&self, dialogue_id: &str) -> Option<usize> {
        let slot_owners = self.slot_owners.lock().await;
        for (&slot_idx, id) in slot_owners.iter() {
            if id == dialogue_id {
                return Some(slot_idx);
            }
        }
        None
    }

    pub async fn is_slot_owned_by(&self, slot_index: usize, dialogue_id: &str) -> bool {
        let slot_owners = self.slot_owners.lock().await;
        slot_owners.get(&slot_index) == Some(&dialogue_id.to_string())
    }

    pub fn available_permits(&self) -> usize {
        self.available_slots.available_permits()
    }
}
