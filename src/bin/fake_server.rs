#![feature(f16)]

use ellm::serving::record::{Phase, SequenceState};
use ellm::common::send_sync_ptr::SharedMut;
use ellm::mem_mgr::allocator::allocate_init;
use ellm::serving::batch_sequence::BatchSequence;
use ellm::serving::server;
use std::sync::Arc;
use std::time::Duration;
use tokenizers::Tokenizer;

fn build_fake_tokens(tokenizer: &Tokenizer) -> Arc<Vec<usize>> {
    let tokens = tokenizer
        .encode("Hello from fake inference.", true)
        .map(|encoded| encoded.get_ids().iter().map(|id| *id as usize).collect())
        .unwrap_or_else(|_| vec![0usize]);
    Arc::new(tokens)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting fake inference server...");

    let sequence_length = 128;
    let batch_size = 4;
    let sequence_capacity = sequence_length + 1;

    let sequences = allocate_init::<usize>(sequence_capacity * batch_size, 0);

    let tokenizer_path = "models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";
    let chat_template_path = "models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";

    let batch_sequences = Arc::new(SharedMut::new(
        BatchSequence::new(
            sequences,
            batch_size,
            sequence_capacity,
            tokenizer_path,
            chat_template_path,
        )
        .map_err(|e| format!("Unable to initialize BatchSequence: {}", e))?,
    ));

    let mut batch_list = Vec::with_capacity(batch_size);
    batch_list.extend((0..batch_size).map(|_| SequenceState {
        sequence_index: 0,
        kv_index: 0,
        phase: Phase::Start,
        // prompt_length: 0,
        notify: Arc::new(tokio::sync::Notify::new()),
    }));
    let batch_list = Arc::new(SharedMut::new(batch_list));

    let tokenizer = {
        let guard = unsafe { &*batch_sequences.get() };
        guard.tokenizer.clone()
    };
    let fake_tokens = build_fake_tokens(&tokenizer);
    let fake_sequences = batch_sequences.clone();
    let fake_batch_list = batch_list.clone();
    std::thread::spawn(move || {
        let max_new_tokens = 16usize;
        let mut slot_prompt_starts = vec![0usize; batch_size];
        let mut slot_active = vec![false; batch_size];
        loop {
            let mut processed = false;
            {
                let batch_list_guard = unsafe { &mut *fake_batch_list.get() };
                let batch_sequences_guard = unsafe { &mut *fake_sequences.get() };
                let capacity = batch_sequences_guard.row_size * batch_sequences_guard.col_size;

                for (slot_index, record) in batch_list_guard.iter_mut().enumerate() {
                    if record.phase != Phase::Decode {
                        slot_active[slot_index] = false;
                        continue;
                    }

                    if !slot_active[slot_index] {
                        slot_prompt_starts[slot_index] = record.sequence_index;
                        slot_active[slot_index] = true;
                    }

                    let generated =
                        record.sequence_index.saturating_sub(slot_prompt_starts[slot_index]);
                    if generated >= max_new_tokens {
                        record.phase = Phase::Eos;
                        record.notify.notify_one();
                        slot_active[slot_index] = false;
                        continue;
                    }

                    let token_id = fake_tokens
                        .get(generated % fake_tokens.len().max(1))
                        .copied()
                        .unwrap_or(0);
                    let out_offset =
                        slot_index * batch_sequences_guard.col_size + record.sequence_index;
                    if out_offset < capacity {
                        unsafe {
                            batch_sequences_guard
                                .sequences
                                .add(out_offset)
                                .write(token_id);
                        }
                    }

                    record.sequence_index = record.sequence_index.saturating_add(1);
                    record.kv_index = record.sequence_index;

                    if record.sequence_index.saturating_sub(slot_prompt_starts[slot_index])
                        >= max_new_tokens
                    {
                        record.phase = Phase::Eos;
                        record.notify.notify_one();
                        slot_active[slot_index] = false;
                    }

                    processed = true;
                }
            }

            if !processed {
                std::thread::sleep(Duration::from_millis(5));
            } else {
                std::thread::yield_now();
            }
        }
    });

    server::run(batch_sequences, batch_list).await?;
    Ok(())
}
