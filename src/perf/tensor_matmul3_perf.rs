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

const SEQUENCE_LENGTH: usize = 1280;
const SEQUENCE_CHUNK_SIZE: usize = 1;
const BATCH_SIZE: usize = 1;
const PADDED_BATCH_SIZE: usize = 3;

const HIDDEN_SIZE: usize = 4096;
const HEAD_DIM: usize = 128;
const NUM_ATTENTION_HEADS: usize = 384;
const NUM_KEY_VALUE_HEADS: usize = 64;
const Q_DIM: usize = NUM_ATTENTION_HEADS * HEAD_DIM;
const KV_DIM: usize = NUM_KEY_VALUE_HEADS * HEAD_DIM;

fn main() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported on this machine, skipping matmul3 perf run.");
        return;
    }

    println!(
        "matmul3 perf init: seq_len={}, seq_chunk={}, batch={}, padded_batch={}, hidden={}, q_dim={}, kv_dim={}, head_dim={}",
        SEQUENCE_LENGTH,
        SEQUENCE_CHUNK_SIZE,
        BATCH_SIZE,
        PADDED_BATCH_SIZE,
        HIDDEN_SIZE,
        Q_DIM,
        KV_DIM,
        HEAD_DIM
    );

    let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
    let operator_queue = Rc::new(RefCell::new(Vec::new()));

    let input_tensor = Tensor::<f16>::from_cache(
        vec![SEQUENCE_CHUNK_SIZE, PADDED_BATCH_SIZE, HIDDEN_SIZE],
        "model.layers.0.perf.matmul3.input".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let q_weight = Tensor::<f16>::from_cache(
        vec![Q_DIM, HIDDEN_SIZE],
        "model.layers.0.perf.matmul3.q_weight".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let k_weight = Tensor::<f16>::from_cache(
        vec![KV_DIM, HIDDEN_SIZE],
        "model.layers.0.perf.matmul3.k_weight".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let v_weight = Tensor::<f16>::from_cache(
        vec![KV_DIM, HIDDEN_SIZE],
        "model.layers.0.perf.matmul3.v_weight".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let position_embedding = Tensor::<f16>::from_cache(
        vec![HEAD_DIM],
        "model.layers.0.perf.matmul3.rope".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    for batch in 0..BATCH_SIZE {
        for kk in 0..HIDDEN_SIZE {
            let value = (((batch * 17 + kk * 13) % 97) as f32 * 0.01) as f16;
            unsafe {
                *input_tensor.data.add(batch * HIDDEN_SIZE + kk) = value;
            }
        }
    }

    for n in 0..Q_DIM {
        for kk in 0..HIDDEN_SIZE {
            let value = (((n * 19 + kk * 7) % 89) as f32 * 0.01) as f16;
            unsafe {
                *q_weight.data.add(n * HIDDEN_SIZE + kk) = value;
            }
        }
    }

    for n in 0..KV_DIM {
        for kk in 0..HIDDEN_SIZE {
            let k_value = (((n * 23 + kk * 5) % 101) as f32 * 0.01) as f16;
            let v_value = (((n * 29 + kk * 11) % 103) as f32 * 0.01) as f16;
            unsafe {
                *k_weight.data.add(n * HIDDEN_SIZE + kk) = k_value;
                *v_weight.data.add(n * HIDDEN_SIZE + kk) = v_value;
            }
        }
    }

    for i in 0..HEAD_DIM {
        let value = (((i * 31) % 113) as f32 * 0.01) as f16;
        unsafe {
            *position_embedding.data.add(i) = value;
        }
    }

    let params = MatMulParams {
        a_row_step_macro: 3,
        b_row_step_macro: 128,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let (_q_state, _k_state, _v_state) = input_tensor.matmul3(
        &q_weight,
        &k_weight,
        &v_weight,
        &position_embedding,
        HEAD_DIM,
        params,
        "model.layers.0.perf.matmul3".to_string(),
    );

    assert_eq!(
        operator_queue.borrow().len(),
        1,
        "expected exactly one operator from matmul3"
    );

    let cpu_num = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let started_at = Instant::now();
    start(operator_queue.take(), SEQUENCE_LENGTH, BATCH_SIZE);
    let elapsed = started_at.elapsed();

    let total_n = Q_DIM + KV_DIM + KV_DIM;
    let flops =
        2.0f64 * SEQUENCE_LENGTH as f64 * BATCH_SIZE as f64 * HIDDEN_SIZE as f64 * total_n as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    println!(
        "matmul3 perf: batch={}, hidden={}, q_dim={}, kv_dim={}, threads={}, elapsed={:?}, effective_gflops={:.2}",
        BATCH_SIZE,
        HIDDEN_SIZE,
        Q_DIM,
        KV_DIM,
        cpu_num,
        elapsed,
        gflops
    );
}
