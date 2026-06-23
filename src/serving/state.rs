use std::sync::Arc;

use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::session::{SessionHandle, SlotManager};
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::shared::SharedState;
use crate::runtime::{Phase, SlotState};

use super::error::{ApiError, ApiResult};
use super::parser::ParserOptions;
use super::requests::ChatMessage;

#[derive(Clone)]
pub struct ApiState {
    pub batch_sequences: Arc<SharedMut<BatchSequence<f16>>>,
    pub batch_states: Arc<SharedMut<Vec<SlotState>>>,
    pub shared_state: Arc<SharedState>,
    pub parser_options: ParserOptions,
    pub slot_manager: Arc<SlotManager<f16>>,
}



impl ApiState {
    /// 将 ChatMessage 转换为消息对数组
    fn messages_to_pairs(messages: &[ChatMessage]) -> Vec<(&str, &str)> {
        messages
            .iter()
            .map(|msg| (msg.role.as_str(), msg.content.as_str()))
            .collect()
    }

    /// 准备 slot 用于写入
    fn prepare_slot_for_write(&self, slot_index: usize) -> ApiResult<()> {
        self.batch_states.with(|batch_list| {
            let record = &batch_list[slot_index];
            if !record.is_available() {
                Err(ApiError::SlotUnavailable(
                    "slot is not in Start or Eos phase".to_string(),
                ))
            } else {
                Ok(())
            }
        })
    }

    pub async fn acquire_session(&self, session_id: &str) -> ApiResult<SessionHandle> {
        self.slot_manager
            .acquire_session(session_id)
            .await
            .map_err(ApiError::from)
    }

    pub async fn release_session(&self, session_id: &str, token_count: usize) {
        self.slot_manager
            .release_session(session_id, token_count)
            .await;
    }

    pub async fn get_cached_prefix(
        &self,
        session_id: &str,
        new_tokens: &[u32],
    ) -> Option<(usize, Vec<u32>)> {
        self.slot_manager
            .calculate_delta(session_id, new_tokens)
            .await
    }

    pub async fn write_prompts_and_prepare(
        &self,
        slot_index: usize,
        messages: &[ChatMessage],
        temperature: Option<f32>,
    ) -> ApiResult<(usize, Arc<tokio::sync::Notify>)> {
        self.prepare_slot_for_write(slot_index)?;

        let message_pairs = Self::messages_to_pairs(messages);
        let temperature = temperature.unwrap_or(1.0);

        let result = self.batch_states.with_mut(|batch_list| {
            self.batch_sequences.with_mut(|batch_sequences| {
                batch_sequences
                    .write_prompts(slot_index, &message_pairs, temperature)
                    .map(|write_len| {
                        let record = &mut batch_list[slot_index];
                        record.sequence_index = 0;
                        record.kv_index = 0;
                        record.filling_length = write_len;
                        record.phase = Phase::Prefill;
                        (write_len, record.notify.clone())
                    })
                    .map_err(|e| ApiError::TokenizationError(e))
            })
        })?;

        self.slot_manager
            .transition_to_prefill(slot_index, 0, result.0)
            .await;

        self.shared_state.push_request();

        Ok(result)
    }

    pub async fn write_prompts_with_incremental_prefill(
        &self,
        slot_index: usize,
        session_id: &str,
        messages: &[ChatMessage],
        temperature: Option<f32>,
    ) -> ApiResult<(usize, Arc<tokio::sync::Notify>)> {
        let message_pairs = Self::messages_to_pairs(messages);

        let new_tokens: Vec<u32> = self.batch_sequences.with(|batch_sequences| {
            batch_sequences
                .tokenize_messages(&message_pairs)
                .unwrap_or_default()
        });

        let result = self.get_cached_prefix(session_id, &new_tokens).await;

        let (write_len, notify) = match result {
            Some((prefix_len, delta_tokens)) => {
                self.prepare_slot_for_write(slot_index)?;
                let temperature = temperature.unwrap_or(1.0);

                self.batch_states.with_mut(|batch_list| {
                    self.batch_sequences.with_mut(|batch_sequences| {
                        batch_sequences
                            .write_tokens(slot_index, &delta_tokens, temperature)
                            .map(|write_len| {
                                let record = &mut batch_list[slot_index];
                                record.sequence_index = prefix_len;
                                record.kv_index = slot_index;
                                record.filling_length = write_len;
                                record.phase = Phase::Prefill;
                                (write_len, record.notify.clone())
                            })
                            .map_err(|e| ApiError::TokenizationError(e))
                    })
                })?
            }
            None => {
                return self
                    .write_prompts_and_prepare(slot_index, messages, temperature)
                    .await;
            }
        };

        self.slot_manager
            .transition_to_prefill(slot_index, 0, write_len)
            .await;

        self.shared_state.push_request();

        Ok((write_len, notify))
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
