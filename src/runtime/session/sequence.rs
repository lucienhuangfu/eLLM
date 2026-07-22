use std::sync::Arc;

use tiktoken_rs::CoreBPE;

use crate::mem_mgr::allocator::AlignedBox;
use crate::num_traits::FromNumber;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::loader::{load_tiktoken, ChatTemplate};
use crate::runtime::session::SlotState;

pub struct BatchSequence<T> {
    pub sequences: *mut usize,
    pub batch_temperature: Vec<T>,
    pub row_size: usize,
    pub col_size: usize,
    pub tokenizer: Arc<CoreBPE>,
    pub chat_template: Arc<ChatTemplate>,
}

impl<T> BatchSequence<T>
where
    T: Copy + FromNumber,
{
    pub fn new(
        sequences: *mut usize,
        row_size: usize,
        col_size: usize,
        tokenizer_json_path: &str,
        tokenizer_config_json_path: &str,
        chat_template_path: &str,
    ) -> Result<Self, String> {
        let tokenizer =
            load_tiktoken(tokenizer_json_path, tokenizer_config_json_path).map(Arc::new)?;
        let chat_template =
            ChatTemplate::from_model_files(chat_template_path, tokenizer_config_json_path)
                .map(Arc::new)
                .map_err(|e| {
                    format!(
                        "Unable to load chat template from {} or {}: {}",
                        chat_template_path, tokenizer_config_json_path, e
                    )
                })?;

        Ok(Self {
            sequences,
            batch_temperature: vec![T::from_f32(1.0); row_size],
            row_size,
            col_size,
            tokenizer,
            chat_template,
        })
    }

    pub fn write_prompts(
        &mut self,
        slot_index: usize,
        messages: &[(&str, &str)],
        temperature: f32,
    ) -> Result<usize, String> {
        let prompt = self
            .chat_template
            .apply_chat_template(messages, true)
            .map_err(|e| format!("Render chat template failed: {}", e))?;
        let tokens = self.tokenizer.encode_with_special_tokens(prompt.as_str());
        let write_len = tokens.len().min(self.col_size);

        let offset = slot_index * self.col_size;

        for (i, id) in tokens[..write_len].iter().enumerate() {
            unsafe {
                *self.sequences.add(offset + i) = *id as usize;
            }
        }

        self.batch_temperature[slot_index] = T::from_f32(temperature);

        println!(
            "Prompt 已通过 tiktoken 写入 BatchSequence Slot {}, 长度: {}, temperature: {}",
            slot_index, write_len, temperature
        );
        Ok(write_len)
    }

    pub fn decode_generated_text(&self, slot_index: usize, record: &SlotState) -> String {
        let sequence_index = record.sequence_index;
        let kv_index = record.kv_index;
        self.decode_token_span(slot_index, kv_index, sequence_index)
    }

    pub fn decode_single_token(&self, slot_index: usize, token_index: usize) -> Option<String> {
        let ids = self.token_ids(slot_index, token_index, token_index + 1);
        if ids.is_empty() {
            return None;
        }
        Some(
            self.tokenizer
                .decode(ids)
                .unwrap_or_else(|_| String::from("?")),
        )
    }

    pub fn decode_token_span(&self, slot_index: usize, begin: usize, end: usize) -> String {
        let token_ids = self.token_ids(slot_index, begin, end);
        if token_ids.is_empty() {
            return String::new();
        }

        self.tokenizer
            .decode(token_ids)
            .unwrap_or_else(|_| String::from("Decode error"))
    }

    pub fn token_ids(&self, slot_index: usize, begin: usize, end: usize) -> Vec<u32> {
        let start = slot_index * self.col_size + begin;
        let end = slot_index * self.col_size + end;
        let capacity = self.row_size * self.col_size;

        if end <= start || end > capacity {
            return Vec::new();
        }

        unsafe {
            let token_slice = std::slice::from_raw_parts(self.sequences.add(start), end - start);
            token_slice.iter().map(|&id| id as u32).collect()
        }
    }

    pub fn tokenize_messages(&self, messages: &[(&str, &str)]) -> Result<Vec<u32>, String> {
        let prompt = self
            .chat_template
            .apply_chat_template(messages, true)
            .map_err(|e| format!("Render chat template failed: {}", e))?;
        let tokens = self.tokenizer.encode_with_special_tokens(prompt.as_str());
        Ok(tokens)
    }

    pub fn write_tokens_at(
        &mut self,
        slot_index: usize,
        start_pos: usize,
        tokens: &[u32],
        temperature: f32,
    ) -> Result<usize, String> {
        let max_write = self.col_size.saturating_sub(start_pos);
        let write_len = tokens.len().min(max_write);

        let offset = slot_index * self.col_size + start_pos;

        for (i, id) in tokens[..write_len].iter().enumerate() {
            unsafe {
                *self.sequences.add(offset + i) = *id as usize;
            }
        }

        self.batch_temperature[slot_index] = T::from_f32(temperature);

        Ok(write_len)
    }
}

unsafe impl<T> Send for BatchSequence<T> where T: Send + Copy + FromNumber {}
unsafe impl<T> Sync for BatchSequence<T> where T: Sync + Copy + FromNumber {}

pub fn build_batch_sequence(
    model_dir: &str,
    batch_size: usize,
    sequence_length: usize,
) -> Result<(AlignedBox<usize>, Arc<SharedMut<BatchSequence<f16>>>), Box<dyn std::error::Error>> {
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tokenizer_config_path = format!("{}/tokenizer_config.json", model_dir);
    let chat_template_path = format!("{}/chat_template.jinja", model_dir);

    let sequences_capacity = sequence_length * batch_size;
    let sequences_box = AlignedBox::allocate_init(sequences_capacity, 0);
    let sequences_ptr = sequences_box.as_mut_ptr();

    let batch_sequences = BatchSequence::<f16>::new(
        sequences_ptr,
        batch_size,
        sequence_length,
        &tokenizer_path,
        &tokenizer_config_path,
        &chat_template_path,
    )
    .map_err(|e| format!("failed to create batch sequence: {}", e))?;

    Ok((sequences_box, Arc::new(SharedMut::new(batch_sequences))))
}
