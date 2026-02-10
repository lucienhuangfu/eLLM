#![feature(f16)]

use ellm::common::record::{BatchRecord, Phase};
use ellm::mem_mgr::allocator::allocate_init;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;
use ellm::serving::schedule::BatchScheduler;
use ellm::serving::runner::ServingRunner;

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

    let eos_id = 151643;

    let (output_indices, output_tensor) = unsafe { model.forward(sequences, eos_id) };

    // let _ = unsafe { model.forward(sequences) };

    let thread_num = core_affinity::get_core_ids().unwrap().len();
    let mut batch_scheduler = BatchScheduler::new(sequence_length, batch_size, thread_num);
    batch_scheduler
        .batch_list
        .extend((0..batch_size).map(|i| BatchRecord {
            sequence_index: i,
            snapshot_sequence_index: 0,
            kv_index: i,
            phase: Phase::Decode,
            prompt_length: i,
            notify: tokio::sync::Notify::new(),
        }));

    ServingRunner::new(model.operator_queue.take(), batch_scheduler).start();
}

