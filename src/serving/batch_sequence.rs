use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::serving::chat_template::ChatTemplate;

pub struct BatchSequence {
    pub sequences: *mut usize, // 展平的二维矩阵 [row_size][col_size]
    pub row_size: usize,
    pub col_size: usize,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Arc<ChatTemplate>,
}

impl BatchSequence {
    pub fn new(
        sequences: *mut usize,
        row_size: usize,
        col_size: usize,
        tokenizer: Arc<Tokenizer>,
        chat_template: Arc<ChatTemplate>,
    ) -> Self {
        Self {
            sequences,
            row_size,
            col_size,
            tokenizer,
            chat_template,
        }
    }

    pub fn write_messages(
        &mut self,
        slot_index: usize,
        messages: &[(&str, &str)],
    ) -> Result<usize, String> {
        let prompt = self
            .chat_template
            .apply_chat_template(messages)
            .map_err(|e| format!("Render chat template failed: {}", e))?;
        let tokens = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        let ids = tokens.get_ids();
        let write_len = ids.len().min(self.col_size);

        // 计算在展平矩阵中的起始偏移量
        let offset = slot_index * self.col_size;

        // 将 tokens 写入对应 slot 的 buffer
        for (i, id) in ids[..write_len].iter().enumerate() {
            unsafe {
                *self.sequences.add(offset + i) = *id as usize;
            }
        }

        println!(
            "Prompt 已通过 Tokenizer 写入 BatchSequence Slot {}, 长度: {}",
            slot_index, write_len
        );
        Ok(write_len)
    }
}

// Raw pointer is shared without locking; callers must ensure safe access.
unsafe impl Send for BatchSequence {}
unsafe impl Sync for BatchSequence {}
