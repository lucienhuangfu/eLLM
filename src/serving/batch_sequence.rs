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
        tokenizer_path: &str,
        chat_template_path: &str,
    ) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map(Arc::new)
            .map_err(|e| format!("Unable to load tokenizer {}: {}", tokenizer_path, e))?;
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

#[cfg(test)]
mod tests {
    use super::BatchSequence;

    const QWEN3_TEMPLATE_PATH: &str = "./models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";
    const QWEN3_TOKENIZER_PATH: &str = "./models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";

    #[test]
    fn test_write_messages_with_qwen3_assets() {
        let mut storage = vec![0usize; 256];
        let mut batch = match BatchSequence::new(
            storage.as_mut_ptr(),
            1,
            storage.len(),
            QWEN3_TOKENIZER_PATH,
            QWEN3_TEMPLATE_PATH,
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skip: qwen3 assets are not loadable in this environment: {}", e);
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
        let expected_ids = batch
            .tokenizer
            .encode(prompt, true)
            .expect("encode failed")
            .get_ids()
            .to_vec();

        let written = batch.write_messages(0, &messages).expect("write failed");
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
    fn test_write_messages_respects_col_size_limit() {
        let mut storage = vec![0usize; 8];
        let mut batch = match BatchSequence::new(
            storage.as_mut_ptr(),
            1,
            storage.len(),
            QWEN3_TOKENIZER_PATH,
            QWEN3_TEMPLATE_PATH,
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skip: qwen3 assets are not loadable in this environment: {}", e);
                return;
            }
        };

        let messages = vec![
            ("system", "You are a helpful assistant."),
            ("user", "请给出一个尽量详细且较长的回答，包含多个步骤和注意事项。"),
        ];

        let written = batch.write_messages(0, &messages).expect("write failed");
        assert_eq!(written, storage.len());
        assert!(storage.iter().any(|id| *id != 0));
    }
}
