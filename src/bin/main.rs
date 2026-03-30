#![feature(f16)]
#![feature(duration_millis_float)]

use ellm::memory::allocator::allocate_init;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;
use ellm::serving::start::start;

fn main() {
    println!("Initializing...");

    let sequence_length = 128;
    let sequence_chunk_size = 1;
    let batch_size = 3;
    let topk_size = 8;

    let config =
        Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config1.json").unwrap();

    let mut model = Model::<f16>::new(
        &config,
        sequence_length,
        sequence_chunk_size,
        batch_size,
        topk_size,
    );

    let sequences = allocate_init::<usize>((sequence_length + 1) * batch_size, 0);
    let _ = model.forward(sequences);

    start(model.operator_queue.take(), sequence_length, batch_size);
}
