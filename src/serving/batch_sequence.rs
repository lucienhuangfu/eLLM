use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct BatchSequence {
    pub sequences: *mut usize, // 展平的二维矩阵 [row_size][col_size]
    pub row_size: usize,
    pub col_size: usize,
}

impl BatchSequence {
    pub fn new(sequences: *mut usize, row_size: usize, col_size: usize) -> Self {
        Self {
            sequences,
            row_size,
            col_size,
        }
    }

    pub fn write_prompt(
        &mut self,
        slot_index: usize,
        prompt: &str,
        tokenizer: &Tokenizer,
    ) -> Result<usize, String> {
        // 使用真正的分词器进行编码
        let tokens = tokenizer
            .encode(prompt, true)
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
