use std::sync::Arc;
use std::time::Duration;

use axum::response::IntoResponse;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::error::SlotError;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::session::{SessionHandle, SessionManager, SessionMode};
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::types::{Phase, SequenceState};

use super::parser::ParserOptions;
use super::requests::ChatMessage;

#[derive(Clone)]
pub struct ApiState<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    pub batch_sequences: Arc<SharedMut<BatchSequence<T>>>,
    pub batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    pub scheduler: Arc<Scheduler>,
    pub parser_options: ParserOptions,
    pub session_manager: Arc<SessionManager<T>>,
}

pub fn build_api_state(
    batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    batch_states: Arc<SharedMut<Vec<SequenceState>>>,
    scheduler: Arc<Scheduler>,
    parser_options: ParserOptions,
    slot_reuse_timeout_ms: usize,
) -> ApiState<f16> {
    let session_manager = Arc::new(SessionManager::new(
        batch_states.clone(),
        batch_sequences.clone(),
        batch_states.with(|states| states.len()),
        Duration::from_millis(slot_reuse_timeout_ms as u64),
    ));

    ApiState {
        batch_sequences,
        batch_states,
        scheduler,
        parser_options,
        session_manager,
    }
}

impl<T> ApiState<T>
where
    T: Copy + crate::num_traits::FromNumber,
{
    /// 获取会话（替代 acquire_slot）
    pub async fn acquire_session(
        &self,
        session_id: &str,
        mode: SessionMode,
    ) -> Result<SessionHandle, axum::response::Response> {
        self.session_manager
            .acquire_session(session_id, mode)
            .await
            .map_err(|e| match e {
                SlotError::AllocatorUnavailable => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Slot allocator unavailable".to_string(),
                )
                    .into_response(),
                SlotError::SlotQueueEmpty => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "No available slots".to_string(),
                )
                    .into_response(),
                SlotError::SlotNotFound => (
                    axum::http::StatusCode::NOT_FOUND,
                    "Slot not found".to_string(),
                )
                    .into_response(),
            })
    }

    /// 释放会话（替代 release_slot）
    pub async fn release_session(&self, session_id: &str, token_count: usize) {
        self.session_manager
            .release_session(session_id, token_count)
            .await;
    }

    /// 检查是否有缓存的 tokens（用于增量预填充）
    pub async fn get_cached_prefix(
        &self,
        session_id: &str,
        new_tokens: &[u32],
    ) -> Option<(usize, Vec<u32>)> {
        self.session_manager
            .calculate_delta(session_id, new_tokens)
            .await
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
        session_id: &str,
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

        let result = self.get_cached_prefix(session_id, &new_tokens).await;

        match result {
            Some((prefix_len, delta_tokens)) => self
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
                                    record.kv_index = slot_index;
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
