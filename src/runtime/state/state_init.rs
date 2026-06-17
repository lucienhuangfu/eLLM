use std::sync::Arc;

use crate::mem_mgr::allocator::AlignedBox;
use crate::operators::send_sync_ptr::SharedMut;
use crate::runtime::state::batch::BatchSequence;
use crate::runtime::state::types::SequenceState;

/// 构建序列状态
///
/// 创建指定数量的初始序列状态
pub fn build_sequence_state(batch_size: usize) -> Vec<SequenceState> {
    (0..batch_size)
        .map(|_| SequenceState {
            filling_length: 0,
            sequence_index: usize::MAX,
            kv_index: usize::MAX,
            phase: crate::runtime::state::types::Phase::Start,
            notify: Arc::new(tokio::sync::Notify::new()),
        })
        .collect()
}

/// 构建批处理序列
///
/// 创建批处理序列和对应的内存分配
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