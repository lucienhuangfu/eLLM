#![feature(f16)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::f16;
use std::rc::Rc;
use std::time::Instant;

use ellm::memory::allocator::allocate_init;
use ellm::memory::cache::Cache;
use ellm::ptensor::tensor::Tensor;
use ellm::serving::start::start;

const SEQUENCE_LENGTH: usize = 1280;
const SEQUENCE_CHUNK_SIZE: usize = 1;
const BATCH_SIZE: usize = 24;
const NUM_EXPERTS: usize = 128;
const TOPK: usize = 8;
const HIDDEN_SIZE: usize = 8192;

fn main() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported on this machine, skipping experts_merge_add perf run.");
        return;
    }

    println!(
        "experts_merge_add perf init: seq_len={}, seq_chunk={}, batch={}, num_experts={}, topk={}, hidden={}",
        SEQUENCE_LENGTH,
        SEQUENCE_CHUNK_SIZE,
        BATCH_SIZE,
        NUM_EXPERTS,
        TOPK,
        HIDDEN_SIZE
    );

    let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
    let operator_queue = Rc::new(RefCell::new(Vec::new()));

    let input_tensor = Tensor::<f16>::from_cache(
        vec![SEQUENCE_CHUNK_SIZE, BATCH_SIZE, TOPK, HIDDEN_SIZE],
        "model.layers.0.perf.experts_merge_add.input".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let residual_tensor = Tensor::<f16>::from_cache(
        vec![SEQUENCE_CHUNK_SIZE, BATCH_SIZE, HIDDEN_SIZE],
        "model.layers.0.perf.experts_merge_add.residual".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let num_tokens = SEQUENCE_CHUNK_SIZE * BATCH_SIZE;
    let experts_indicator = unsafe { allocate_init(NUM_EXPERTS, false) };
    let indice_ptr = unsafe { allocate_init(NUM_EXPERTS * num_tokens, false) };

    unsafe {
        for e in 0..NUM_EXPERTS {
            *experts_indicator.add(e) = true;
        }
        for i in 0..(NUM_EXPERTS * num_tokens) {
            *indice_ptr.add(i) = true;
        }
    }

    for t in 0..num_tokens {
        for h in 0..HIDDEN_SIZE {
            let value = (((t * 17 + h * 13) % 97) as f32 * 0.01) as f16;
            unsafe {
                *residual_tensor.data.add(t * HIDDEN_SIZE + h) = value;
            }
        }
    }

    for t in 0..num_tokens {
        let token_base = t * TOPK * HIDDEN_SIZE;
        for s in 0..TOPK {
            let slot_base = token_base + s * HIDDEN_SIZE;
            for h in 0..HIDDEN_SIZE {
                let value = (((t * 19 + s * 23 + h * 7) % 101) as f32 * 0.01) as f16;
                unsafe {
                    *input_tensor.data.add(slot_base + h) = value;
                }
            }
        }
    }

    let _output = input_tensor.experts_merge_add(
        &residual_tensor,
        experts_indicator,
        indice_ptr,
        NUM_EXPERTS,
        "model.layers.0.perf.experts_merge_add".to_string(),
    );

    assert_eq!(
        operator_queue.borrow().len(),
        1,
        "expected exactly one operator from experts_merge_add"
    );

    let cpu_num = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let started_at = Instant::now();
    start(operator_queue.take(), SEQUENCE_LENGTH, BATCH_SIZE);
    let elapsed = started_at.elapsed();

    let residual_copy_ops =
        SEQUENCE_LENGTH as f64 * BATCH_SIZE as f64 * HIDDEN_SIZE as f64;
    let merge_add_ops =
        SEQUENCE_LENGTH as f64 * BATCH_SIZE as f64 * TOPK as f64 * HIDDEN_SIZE as f64;
    let effective_gops = (residual_copy_ops + merge_add_ops) / elapsed.as_secs_f64() / 1e9;

    println!(
        "experts_merge_add perf: batch={}, topk={}, hidden={}, threads={}, elapsed={:?}, effective_gops={:.2}",
        BATCH_SIZE,
        TOPK,
        HIDDEN_SIZE,
        cpu_num,
        elapsed,
        effective_gops
    );
}
