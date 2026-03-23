#![feature(f16)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::f16;
use std::rc::Rc;
use std::time::Instant;

use ellm::init::matmul_params::MatMulParams;
use ellm::memory::cache::Cache;
use ellm::ptensor::tensor::Tensor;
use ellm::serving::start::start;

const SEQUENCE_LENGTH: usize = 128;
const SEQUENCE_CHUNK_SIZE: usize = 1;
const BATCH_SIZE: usize = 1;
const K: usize = 2048;
const N: usize = 151936;

fn main() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported on this machine, skipping Qwen3 vocab matmul perf run.");
        return;
    }

    println!(
        "Qwen3 vocab matmul perf init: seq_len={}, seq_chunk={}, batch={}, vocab={}, hidden={}",
        SEQUENCE_LENGTH, SEQUENCE_CHUNK_SIZE, BATCH_SIZE, N, K
    );

    let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
    let operator_queue = Rc::new(RefCell::new(Vec::new()));

    let input_tensor = Tensor::<f16>::from_cache(
        vec![SEQUENCE_CHUNK_SIZE, BATCH_SIZE, K],
        "perf.qwen3_vocab.input".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let weight_tensor = Tensor::<f16>::from_cache(
        vec![N, K],
        "perf.qwen3_vocab.weight".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    for batch in 0..BATCH_SIZE {
        for kk in 0..K {
            let value = (((batch * 17 + kk * 13) % 97) as f32 * 0.01) as f16;
            unsafe {
                *input_tensor.data.add(batch * K + kk) = value;
            }
        }
    }

    for n in 0..N {
        for kk in 0..K {
            let value = (((n * 19 + kk * 7) % 89) as f32 * 0.01) as f16;
            unsafe {
                *weight_tensor.data.add(n * K + kk) = value;
            }
        }
    }

    let params = MatMulParams {
        a_row_step_macro: 3,
        b_row_step_macro: 64,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let _output_tensor = input_tensor.matmul(
        &weight_tensor,
        params,
        SEQUENCE_CHUNK_SIZE,
        "perf.qwen3_vocab".to_string(),
    );

    assert_eq!(
        operator_queue.borrow().len(),
        1,
        "expected exactly one operator from Qwen3 vocab matmul"
    );

    let cpu_num = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let started_at = Instant::now();
    start(operator_queue.take(), SEQUENCE_LENGTH, BATCH_SIZE);
    let elapsed = started_at.elapsed();

    let flops = 2.0f64 * SEQUENCE_LENGTH as f64 * BATCH_SIZE as f64 * N as f64 * K as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    println!(
        "Qwen3 vocab matmul perf: batch={}, vocab={}, hidden={}, threads={}, elapsed={:?}, gflops={:.2}",
        BATCH_SIZE,
        N,
        K,
        cpu_num,
        elapsed,
        gflops
    );
}
