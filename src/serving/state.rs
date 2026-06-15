use std::sync::Arc;
use std::time::Duration;

use axum::response::IntoResponse;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::SlotError;
use crate::runtime::scheduling::batch_sequence::BatchSequence;
use crate::runtime::scheduling::{Phase, Scheduler, SequenceState};
use crate::runtime::DialogueCache;
use crate::runtime::SlotManager;

use super::parser::ParserOptions;
use super::requests::ChatMessage;

#[derive(Clone)]
pub struct ApiState {
    pub batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    pub batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    pub scheduler: Arc<Scheduler>,
    pub parser_options: ParserOptions,
    pub slot_manager: Arc<SlotManager>,
    pub dialogue_cache: Arc<DialogueCache>,
}

pub fn build_api_state(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    scheduler: Arc<Scheduler>,
    parser_options: ParserOptions,
) -> ApiState {
    let slot_manager = Arc::new(SlotManager::new(batch_states.clone()));
    let dialogue_cache = Arc::new(DialogueCache::new(
        slot_manager.clone(),
        batch_sequences.clone(),
        Duration::from_secs(10),
        batch_states.with(|states| states.len()),
    ));

    ApiState {
        batch_sequences,
        batch_states,
        scheduler,
        parser_options,
        slot_manager,
        dialogue_cache,
    }
}

impl ApiState {
    pub async fn acquire_slot(&self) -> Result<usize, axum::response::Response> {
        self.slot_manager
            .acquire_slot(None)
            .await
            .map_err(|e| match e {
                SlotError::AllocatorUnavailable => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Slot allocator unavailable".to_string(),
                )
                    .into_response(),
                SlotError::SlotQueueEmpty => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Slot queue empty while permit acquired".to_string(),
                )
                    .into_response(),
                SlotError::SlotNotFound => (
                    axum::http::StatusCode::NOT_FOUND,
                    "Slot not found".to_string(),
                )
                    .into_response(),
            })
    }

    pub async fn acquire_slot_for_dialogue(
        &self,
        dialogue_id: &str,
    ) -> Result<usize, axum::response::Response> {
        self.slot_manager
            .acquire_slot(Some(dialogue_id))
            .await
            .map_err(|e| match e {
                SlotError::AllocatorUnavailable => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Slot allocator unavailable".to_string(),
                )
                    .into_response(),
                SlotError::SlotQueueEmpty => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Slot queue empty while permit acquired".to_string(),
                )
                    .into_response(),
                SlotError::SlotNotFound => (
                    axum::http::StatusCode::NOT_FOUND,
                    "Slot not found".to_string(),
                )
                    .into_response(),
            })
    }

    pub async fn release_slot(&self, slot_index: usize, release_permit: bool) {
        self.slot_manager
            .release_slot(slot_index, release_permit)
            .await;
    }

    pub async fn release_by_dialogue(&self, dialogue_id: &str) -> Option<usize> {
        self.slot_manager.release_by_dialogue(dialogue_id).await
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
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Tokenization failed: {}", err),
                )
                    .into_response()
            })
    }

    pub async fn write_prompts_with_incremental_prefill(
        &self,
        slot_index: usize,
        dialogue_id: &str,
        messages: &[ChatMessage],
        temperature: Option<f32>,
    ) -> Result<(usize, Arc<tokio::sync::Notify>), axum::response::Response> {
        let message_pairs = messages
            .iter()
            .map(|msg| (msg.role.as_str(), msg.content.as_str()))
            .collect::<Vec<_>>();

        let new_tokens: Vec<u32> = self.batch_sequences.with(|batch_sequences| {
            batch_sequences
                .tokenize_messages(&message_pairs)
                .unwrap_or_default()
        });

        let result = self
            .dialogue_cache
            .find_common_prefix(dialogue_id, &new_tokens)
            .await;

        match result {
            Some((entry, prefix_len, delta_tokens)) => self
                .batch_states
                .with_mut(|batch_list| {
                    self.batch_sequences.with_mut(|batch_sequences| {
                        let record = &mut batch_list[slot_index];

                        if !record.is_available() {
                            Err("slot is not in Start or Eos phase".to_string())
                        } else {
                            let temperature = temperature.unwrap_or(1.0);
                            batch_sequences
                                .write_tokens(slot_index, &delta_tokens, temperature)
                                .map(|write_len| {
                                    record.sequence_index = prefix_len;
                                    record.kv_index = entry.slot_index;
                                    record.filling_length = write_len;
                                    record.phase = Phase::Prefill;
                                    (write_len, record.notify.clone())
                                })
                                .map_err(|e| e.to_string())
                        }
                    })
                })
                .map_err(|err| {
                    eprintln!("Error writing incremental prompt: {}", err);
                    (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Tokenization failed: {}", err),
                    )
                        .into_response()
                }),
            None => {
                let result = self.write_prompts_and_prepare(slot_index, messages, temperature)?;

                self.dialogue_cache
                    .insert(dialogue_id.to_string(), slot_index, new_tokens.len())
                    .await;

                Ok(result)
            }
        }
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
