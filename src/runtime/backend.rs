use std::sync::Arc;
use tokio::sync::Notify;

use crate::serving::{ApiError, ApiResult};
use crate::serving::parser::ParserOptions;
use crate::serving::ChatMessage;

use super::scheduler::Scheduler;
use super::session::{Phase, SessionHandle, SlotManager};

#[derive(Clone)]
pub struct Backend {
    pub scheduler: Arc<Scheduler>,
    pub slot_manager: Arc<SlotManager<f16>>,
    pub parser_options: ParserOptions,
}

impl Backend {
    pub fn new(
        scheduler: Arc<Scheduler>,
        slot_manager: Arc<SlotManager<f16>>,
        parser_options: ParserOptions,
    ) -> Self {
        Self {
            scheduler,
            slot_manager,
            parser_options,
        }
    }

    fn messages_to_pairs(messages: &[ChatMessage]) -> Vec<(&str, &str)> {
        messages
            .iter()
            .map(|msg| (msg.role.as_str(), msg.content.as_str()))
            .collect()
    }

    fn prepare_slot_for_write(&self, slot_index: usize) -> ApiResult<()> {
        self.slot_manager.with_slots(|slots| {
            let record = &slots[slot_index];
            if !record.is_available() {
                Err(ApiError::SlotUnavailable(
                    "slot is not in Start or Eos phase".to_string(),
                ))
            } else {
                Ok(())
            }
        })
    }

    async fn get_cached_prefix(&self, session_id: &str, new_tokens: &[u32]) -> Option<usize> {
        self.slot_manager
            .get_prefix_match_len(session_id, new_tokens)
            .await
    }

    pub fn acquire_session<'a>(
        &'a self,
        session_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ApiResult<SessionHandle>> + Send + 'a>>
    {
        Box::pin(async move {
            self.slot_manager
                .acquire_session(session_id)
                .await
                .map_err(ApiError::from)
        })
    }

    pub fn release_session<'a>(
        &'a self,
        session_id: &'a str,
        token_count: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            self.slot_manager
                .release_session(session_id, token_count)
                .await;
        })
    }

    pub fn write_prompts_and_prepare<'a>(
        &'a self,
        slot_index: usize,
        messages: &'a [ChatMessage],
        temperature: Option<f32>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ApiResult<(usize, Arc<Notify>)>> + Send + 'a>>
    {
        Box::pin(async move {
            self.prepare_slot_for_write(slot_index)?;
            self.slot_manager.detach_from_lru(slot_index);

            let message_pairs = Self::messages_to_pairs(messages);
            let temperature = temperature.unwrap_or(1.0);

            let result = self
                .slot_manager
                .with_slot_and_sequence_mut(slot_index, |record, seq| {
                    seq.write_prompts(slot_index, &message_pairs, temperature)
                        .map(|write_len| {
                            record.sequence_index = 0;
                            record.kv_index = 0;
                            record.filling_length = write_len;
                            record.phase = Phase::Prefill;
                            (write_len, record.notify.clone())
                        })
                        .map_err(|e| ApiError::TokenizationError(e))
                })?;

            Ok(result)
        })
    }

    pub fn write_prompts_with_incremental_prefill<'a>(
        &'a self,
        slot_index: usize,
        session_id: &'a str,
        messages: &'a [ChatMessage],
        temperature: Option<f32>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ApiResult<(usize, Arc<Notify>)>> + Send + 'a>>
    {
        Box::pin(async move {
            let message_pairs = Self::messages_to_pairs(messages);

            let new_tokens: Vec<u32> = self.slot_manager.with_sequence(|seq| {
                seq.tokenize_messages(&message_pairs)
                    .unwrap_or_default()
            });

            let result = self.get_cached_prefix(session_id, &new_tokens).await;

            let (write_len, notify) = match result {
                Some(prefix_len) => {
                    self.prepare_slot_for_write(slot_index)?;
                    self.slot_manager.detach_from_lru(slot_index);
                    let temperature = temperature.unwrap_or(1.0);
                    let remaining_tokens = &new_tokens[prefix_len..];

                    self.slot_manager
                        .with_slot_and_sequence_mut(slot_index, |record, seq| {
                            seq.write_tokens_at(slot_index, prefix_len, remaining_tokens, temperature)
                                .map(|write_len| {
                                    record.sequence_index = prefix_len;
                                    record.kv_index = slot_index;
                                    record.filling_length = write_len;
                                    record.phase = Phase::Prefill;
                                    (write_len, record.notify.clone())
                                })
                                .map_err(|e| ApiError::TokenizationError(e))
                        })?
                }
                None => {
                    return self
                        .write_prompts_and_prepare(slot_index, messages, temperature)
                        .await;
                }
            };

            Ok((write_len, notify))
        })
    }

    pub fn is_eos(&self, slot_index: usize) -> bool {
        self.slot_manager
            .with_slots(|slots| matches!(slots[slot_index].phase, Phase::Eos))
    }

    pub fn get_token_index_and_phase(&self, slot_index: usize) -> (usize, Phase) {
        self.slot_manager
            .with_slots(|slots| (slots[slot_index].sequence_index, slots[slot_index].phase))
    }

    pub fn decode_single_token(&self, slot_index: usize, token_index: usize) -> String {
        self.slot_manager.decode_single_token(slot_index, token_index)
    }

    pub fn decode_generated_text(&self, slot_index: usize) -> String {
        self.slot_manager.decode_generated_text(slot_index)
    }

    pub fn get_sequence_index(&self, slot_index: usize) -> usize {
        self.slot_manager
            .with_slots(|slots| slots[slot_index].sequence_index)
    }
}
