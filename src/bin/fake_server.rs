#![feature(f16)]

use ellm::common::send_sync_ptr::SharedMut;
use ellm::mem_mgr::allocator::allocate_init;
use ellm::runtime::inference::{Phase, SequenceState};
use ellm::serving::batch_sequence::BatchSequence;
use ellm::serving::server;
use std::sync::Arc;
use std::time::Duration;
use tiktoken_rs::CoreBPE;

fn build_fake_tokens(tokenizer: &CoreBPE) -> Arc<Vec<usize>> {
    let tokens = tokenizer
        .encode_with_special_tokens("Hello from fake inference.")
        .into_iter()
        .map(|id| id as usize)
        .collect();
    Arc::new(tokens)
}

fn start_fake_inference_loop(
    batch_sequences: Arc<SharedMut<BatchSequence>>,
    batch_list: Arc<SharedMut<Vec<SequenceState>>>,
    fake_tokens: Arc<Vec<usize>>,
    batch_size: usize,
) {
    std::thread::spawn(move || {
        let max_new_tokens = 24usize;
        let mut prompt_start = vec![0usize; batch_size];

        loop {
            let mut progressed = false;

            batch_sequences.with_mut(|batch_sequences_guard| {
                batch_list.with_mut(|batch_list_guard| {
                    let row_stride = batch_sequences_guard.col_size;
                    let total_capacity = batch_sequences_guard.row_size * row_stride;

                    for (slot_index, record) in batch_list_guard.iter_mut().enumerate() {
                        match record.phase {
                            Phase::Prefill => {
                                let prompt_end = record.kv_index.max(
                                    record.sequence_index.saturating_add(record.filling_length),
                                );
                                prompt_start[slot_index] = prompt_end;
                                record.sequence_index = prompt_end;
                                record.kv_index = prompt_end;
                                record.filling_length = 0;
                                record.phase = Phase::Decode;
                                progressed = true;
                            }
                            Phase::Decode => {
                                let generated =
                                    record.kv_index.saturating_sub(prompt_start[slot_index]);
                                if generated >= max_new_tokens {
                                    record.phase = Phase::Eos;
                                    record.notify.notify_one();
                                    progressed = true;
                                    continue;
                                }

                                let write_pos = slot_index * row_stride + record.kv_index;
                                if write_pos >= total_capacity {
                                    record.phase = Phase::Eos;
                                    record.notify.notify_one();
                                    progressed = true;
                                    continue;
                                }

                                let token_id = fake_tokens
                                    .get(generated % fake_tokens.len().max(1))
                                    .copied()
                                    .unwrap_or(0);

                                unsafe {
                                    batch_sequences_guard
                                        .sequences
                                        .add(write_pos)
                                        .write(token_id);
                                }

                                record.kv_index = record.kv_index.saturating_add(1);

                                if record.kv_index.saturating_sub(prompt_start[slot_index])
                                    >= max_new_tokens
                                {
                                    record.phase = Phase::Eos;
                                    record.notify.notify_one();
                                }

                                progressed = true;
                            }
                            _ => {}
                        }
                    }
                })
            });

            if progressed {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    });
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting fake inference server...");

    let sequence_length = 256usize;
    let batch_size = 4usize;
    let sequence_capacity = sequence_length + 1;

    let sequences = allocate_init::<usize>(sequence_capacity * batch_size, 0);

    let tokenizer_path = "models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json";
    let tokenizer_config_path = "models/Qwen3-Coder-30B-A3B-Instruct/tokenizer_config.json";
    let chat_template_path = "models/Qwen3-Coder-30B-A3B-Instruct/chat_template.jinja";

    let batch_sequences = Arc::new(SharedMut::new(
        BatchSequence::new(
            sequences,
            batch_size,
            sequence_capacity,
            tokenizer_path,
            tokenizer_config_path,
            chat_template_path,
        )
        .map_err(|e| format!("Unable to initialize BatchSequence: {}", e))?,
    ));

    let mut initial_states = Vec::with_capacity(batch_size);
    initial_states.extend((0..batch_size).map(|_| SequenceState {
        filling_length: 0,
        sequence_index: 0,
        kv_index: 0,
        phase: Phase::Start,
        notify: Arc::new(tokio::sync::Notify::new()),
    }));
    let batch_list = Arc::new(SharedMut::new(initial_states));

    let tokenizer = batch_sequences.with(|guard| guard.tokenizer.clone());
    let fake_tokens = build_fake_tokens(&tokenizer);

    start_fake_inference_loop(
        batch_sequences.clone(),
        batch_list.clone(),
        fake_tokens,
        batch_size,
    );

    server::run(batch_sequences, batch_list).await?;
    Ok(())
}
