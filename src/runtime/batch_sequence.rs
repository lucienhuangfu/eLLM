use std::sync::Arc;
use tiktoken_rs::CoreBPE;

use crate::runtime::chat_template::ChatTemplate;
use crate::runtime::tokenizer_loader::load_tiktoken;
use crate::runtime::SequenceState;

pub struct BatchSequence {
    pub sequences: *mut usize,
    pub row_size: usize,
    pub col_size: usize,
    pub tokenizer: Arc<CoreBPE>,
    pub chat_template: Arc<ChatTemplate>,
}

impl BatchSequence {
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
        let chat_template = ChatTemplate::new(chat_template_path)
            .map(Arc::new)
            .map_err(|e| format!("Unable to load chat template {}: {}", chat_template_path, e))?;

        Ok(Self {
            sequences,
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
    ) -> Result<usize, String> {
        let prompt = self
            .chat_template
            .apply_chat_template(messages)
            .map_err(|e| format!("Render chat template failed: {}", e))?;
        let tokens = self.tokenizer.encode_with_special_tokens(prompt.as_str());
        let ids = tokens;
        let write_len = ids.len().min(self.col_size);

        let offset = slot_index * self.col_size;

        for (i, id) in ids[..write_len].iter().enumerate() {
            unsafe {
                *self.sequences.add(offset + i) = *id as usize;
            }
        }

        println!(
            "Prompt 已通过 tiktoken 写入 BatchSequence Slot {}, 长度: {}",
            slot_index, write_len
        );
        Ok(write_len)
    }

    pub fn decode_generated_text(&self, slot_index: usize, record: &SequenceState) -> String {
        let sequence_index = record.sequence_index;
        let kv_index = record.kv_index;
        let start = slot_index * self.col_size + sequence_index;
        let end = slot_index * self.col_size + kv_index;
        let capacity = self.row_size * self.col_size;

        if end <= start || end > capacity {
            return String::new();
        }

        let token_ids: Vec<u32> = unsafe {
            let token_slice = std::slice::from_raw_parts(self.sequences.add(start), end - start);
            token_slice.iter().map(|&id| id as u32).collect()
        };

        self.tokenizer
            .decode(token_ids)
            .unwrap_or_else(|_| String::from("Decode error"))
    }
}

unsafe impl Send for BatchSequence {}
unsafe impl Sync for BatchSequence {}

#[cfg(test)]
mod tests {
    use super::BatchSequence;

    const QWEN3_TEMPLATE_PATH: &str = "./models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";
    const QWEN3_TOKENIZER_PATH: &str = "./models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";
    const QWEN3_TOKENIZER_CONFIG_PATH: &str =
        "./models/Qwen3-Coder-30B-A3B-Instruct/tokenizer_config.json";

    #[test]
    fn test_write_prompts_with_qwen3_assets() {
        let mut storage = vec![0usize; 256];
        let mut batch = match BatchSequence::new(
            storage.as_mut_ptr(),
            1,
            storage.len(),
            QWEN3_TOKENIZER_PATH,
            QWEN3_TOKENIZER_CONFIG_PATH,
            QWEN3_TEMPLATE_PATH,
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "Skip: qwen3 assets are not loadable in this environment: {}",
                    e
                );
                return;
            }
        };

        let messages = vec![
            ("system", "You are a helpful assistant."),
            ("user", "请简要解释 Rust 的所有权。"),
        ];

        let prompt = batch
            .chat_template
            .apply_chat_template(&messages)
            .expect("render failed");
        let expected_ids = batch.tokenizer.encode_with_special_tokens(prompt.as_str());

        let written = batch.write_prompts(0, &messages).expect("write failed");
        assert!(written > 0);
        assert_eq!(written, expected_ids.len().min(storage.len()));
        assert_eq!(
            &storage[..written],
            &expected_ids[..written]
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_write_prompts_respects_col_size_limit() {
        let mut storage = vec![0usize; 8];
        let mut batch = match BatchSequence::new(
            storage.as_mut_ptr(),
            1,
            storage.len(),
            QWEN3_TOKENIZER_PATH,
            QWEN3_TOKENIZER_CONFIG_PATH,
            QWEN3_TEMPLATE_PATH,
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "Skip: qwen3 assets are not loadable in this environment: {}",
                    e
                );
                return;
            }
        };

        let messages = vec![
            ("system", "You are a helpful assistant."),
            (
                "user",
                "请给出一个尽量详细且较长的回答，包含多个步骤和注意事项。",
            ),
        ];

        let written = batch.write_prompts(0, &messages).expect("write failed");
        assert_eq!(written, storage.len());
        assert!(storage.iter().any(|id| *id != 0));
    }
}
