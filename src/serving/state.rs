use std::collections::VecDeque;
use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use tokio::sync::{Mutex, Semaphore};

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::batch_sequence::BatchSequence;
use crate::runtime::scheduling::{Phase, Scheduler, SequenceState};

use super::parser::ParserOptions;
use super::requests::ChatMessage;

#[derive(Clone)]
pub struct ApiState {
    pub batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    pub batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    pub scheduler: Arc<Scheduler>,
    pub parser_options: ParserOptions,
    pub free_slots: Arc<Mutex<VecDeque<usize>>>,
    pub available_slots: Arc<Semaphore>,
}

pub fn build_api_state(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    scheduler: Arc<Scheduler>,
    parser_options: ParserOptions,
) -> ApiState {
    let initial_free_slots: VecDeque<usize> = batch_states.with(|batch_states_ref| {
        batch_states_ref
            .iter()
            .enumerate()
            .filter_map(|(i, record)| (record.phase == Phase::Start).then_some(i))
            .collect()
    });
    let initial_permits = initial_free_slots.len();

    ApiState {
        batch_sequences,
        batch_states,
        scheduler,
        parser_options,
        free_slots: Arc::new(Mutex::new(initial_free_slots)),
        available_slots: Arc::new(Semaphore::new(initial_permits)),
    }
}

impl ApiState {
    pub async fn acquire_slot(&self) -> Result<usize, axum::response::Response> {
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

        permit.forget();
        Ok(slot_index)
    }

    pub async fn release_slot(&self, slot_index: usize, release_permit: bool) {
        self.batch_states.with_mut(|batch_list| {
            if let Some(record) = batch_list.get_mut(slot_index) {
                record.reset_to_start();
            }
        });

        let mut free_slots = self.free_slots.lock().await;
        free_slots.push_back(slot_index);
        drop(free_slots);

        if release_permit {
            self.available_slots.add_permits(1);
        }
    }

    pub fn write_prompts_and_prepare(
        &self,
        slot_index: usize,
        messages: &[ChatMessage],
        temperature: Option<f32>,
    ) -> Result<(usize, Arc<tokio::sync::Notify>), axum::response::Response> {
        let message_pairs = messages
            .iter()
            .map(|msg| (msg.role.as_str(), msg.content.as_str()))
            .collect::<Vec<_>>();

        self.batch_states
            .with_mut(|batch_list| {
                self.batch_sequences.with_mut(|batch_sequences| {
                    let record = &mut batch_list[slot_index];
                    if !record.is_available() {
                        Err("slot is not in Start or Eos phase".to_string())
                    } else {
                        let temperature = temperature.unwrap_or(1.0);
                        batch_sequences
                            .write_prompts(slot_index, &message_pairs, temperature)
                            .map(|write_len| {
                                record.sequence_index = 0;
                                record.kv_index = 0;
                                record.filling_length = write_len;
                                record.phase = Phase::Prefill;
                                (write_len, record.notify.clone())
                            })
                            .map_err(|e| e.to_string())
                    }
                })
            })
            .map_err(|err| {
                eprintln!("Error writing prompt: {}", err);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Tokenization failed: {}", err),
                )
                    .into_response()
            })
    }

    pub fn is_eos(&self, slot_index: usize) -> bool {
        self.batch_states.with(|batch_list| {
            let record = &batch_list[slot_index];
            matches!(record.phase, Phase::Eos)
        })
    }

    pub fn get_token_index_and_phase(&self, slot_index: usize) -> (usize, Phase) {
        self.batch_states.with(|batch_list| {
            let record = &batch_list[slot_index];
            (record.sequence_index, record.phase)
        })
    }

    pub fn decode_single_token(&self, slot_index: usize, token_index: usize) -> String {
        self.batch_sequences.with(|batch_sequences| {
            batch_sequences
                .decode_single_token(slot_index, token_index)
                .unwrap_or_default()
        })
    }

    pub fn decode_generated_text(&self, slot_index: usize) -> String {
        self.batch_states.with(|batch_list| {
            let record = &batch_list[slot_index];
            self.batch_sequences
                .with(|batch_sequences| batch_sequences.decode_generated_text(slot_index, record))
        })
    }
}
