#![feature(f16)]
#![feature(duration_millis_float)]

use ellm::init::record::{BatchRecord, Phase, PrefillEndRecord, TokenList, TokenRecord};
use ellm::init::send_sync_ptr::MutPtr;
use ellm::memory::allocator::allocate_init;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;
use ellm::serving::start::ServingRunner;

fn main() {
    println!("Initializing...");

    let sequence_length = 128;
    let sequence_chunk_size = 1;
    let batch_size = 3;
    let topk_size = 8;

    let config =
        Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json").unwrap();

    let mut model = Model::<f16>::new(&config, sequence_length, batch_size, topk_size);

    let sequences = allocate_init::<usize>((sequence_length + 1) * batch_size, 0);

    let token_records: Vec<TokenRecord> = (0..batch_size)
        .map(|i| TokenRecord {
            // token_id: 0,
            batch_index: i,
            position_index: 0,
        })
        .collect();

    let lift_records: Vec<PrefillEndRecord> = Vec::new();

    let mut token_list = TokenList {
        token_records: token_records.into_boxed_slice(),
        current_token_size: batch_size,
        lift_records: lift_records.into_boxed_slice(),
        current_lift_size: 0,
    };

    let batch_records: Vec<BatchRecord> = (0..batch_size)
        .map(|i| BatchRecord {
            sequence_index: i,
            snapshot_sequence_index: 0,
            kv_index: i,
            phase: Phase::Decode,
            prompt_length: i,
            notify: std::sync::Arc::new(tokio::sync::Notify::new()),
        })
        .collect();

    let mut batch_list = batch_records;
    let eos_id = 151643;

    let (output_indices, output_tensor) = unsafe {
        model.forward(
            sequences,
            &token_list as *const TokenList,
            eos_id,
        )
    };

    // let _ = unsafe { model.forward(sequences) };

    let batch_ptr = MutPtr {
        ptr: &mut batch_list as *mut Vec<BatchRecord>,
    };

    ServingRunner::new(batch_ptr, model.operator_queue.take()).start();
}
