#![feature(f16)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::f16;
use std::rc::Rc;
use std::time::Instant;

use ellm::init::matmul_params::MatMulParams;
use ellm::memory::allocator::allocate_init;
use ellm::memory::cache::Cache;
use ellm::ptensor::tensor::Tensor;
use ellm::serving::start::start;

const SEQUENCE_LENGTH: usize = 1280;
const SEQUENCE_CHUNK_SIZE: usize = 1;
const BATCH_SIZE: usize = 24;

const NUM_EXPERTS: usize = 8;
const TOPK: usize = 2;
const INTERMEDIATE_SIZE: usize = 4096;
const HIDDEN_SIZE: usize = 2048;

fn main() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported on this machine, skipping experts_matmul_mul perf run.");
        return;
    }

    println!(
        "experts_matmul_mul perf init: seq_len={}, seq_chunk={}, batch={}, experts={}, topk={}, inter={}, hidden={}",
        SEQUENCE_LENGTH,
        SEQUENCE_CHUNK_SIZE,
        BATCH_SIZE,
        NUM_EXPERTS,
        TOPK,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE
    );

    let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
    let operator_queue = Rc::new(RefCell::new(Vec::new()));

    let input_tensor = Tensor::<f16>::from_cache(
        vec![NUM_EXPERTS, SEQUENCE_CHUNK_SIZE, BATCH_SIZE, INTERMEDIATE_SIZE],
        "model.layers.0.perf.experts_down.input".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let down_weight = Tensor::<f16>::from_cache(
        vec![NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE],
        "model.layers.0.perf.experts_down.weight".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let num_tokens = SEQUENCE_CHUNK_SIZE * BATCH_SIZE;
    let experts_indicator = allocate_init(NUM_EXPERTS, false);
    let indice_ptr = allocate_init(NUM_EXPERTS * num_tokens, false);
    let weight_ptr = allocate_init(NUM_EXPERTS * num_tokens, 0.0f16);
    let topk_indices_ptr = allocate_init(num_tokens * TOPK, 0usize);

    let mut routed_pairs = 0usize;
    unsafe {
        for t in 0..num_tokens {
            for slot in 0..TOPK {
                let expert = (t * TOPK + slot) % NUM_EXPERTS;
                *experts_indicator.add(expert) = true;
                *indice_ptr.add(expert * num_tokens + t) = true;
                *weight_ptr.add(expert * num_tokens + t) = (1.0f32 / TOPK as f32) as f16;
                *topk_indices_ptr.add(t * TOPK + slot) = expert;
                routed_pairs += 1;
            }
        }
    }

    let input_stride_per_expert = num_tokens * INTERMEDIATE_SIZE;
    for e in 0..NUM_EXPERTS {
        let expert_base = e * input_stride_per_expert;
        for t in 0..num_tokens {
            let token_base = expert_base + t * INTERMEDIATE_SIZE;
            for kk in 0..INTERMEDIATE_SIZE {
                let value = (((e * 17 + t * 13 + kk * 7) % 97) as f32 * 0.01) as f16;
                unsafe {
                    *input_tensor.data.add(token_base + kk) = value;
                }
            }
        }
    }

    let weight_stride_per_expert = HIDDEN_SIZE * INTERMEDIATE_SIZE;
    for e in 0..NUM_EXPERTS {
        let expert_base = e * weight_stride_per_expert;
        for j in 0..HIDDEN_SIZE {
            let row_base = expert_base + j * INTERMEDIATE_SIZE;
            for kk in 0..INTERMEDIATE_SIZE {
                let value = (((e * 19 + j * 11 + kk * 5) % 101) as f32 * 0.01) as f16;
                unsafe {
                    *down_weight.data.add(row_base + kk) = value;
                }
            }
        }
    }

    let params = MatMulParams {
        a_row_step_macro: 6,
        b_row_step_macro: 128,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let _output = input_tensor.experts_matmul_mul(
        &down_weight,
        experts_indicator,
        indice_ptr,
        weight_ptr,
        topk_indices_ptr,
        TOPK,
        params,
        "model.layers.0.perf.experts_down".to_string(),
    );

    assert_eq!(
        operator_queue.borrow().len(),
        1,
        "expected exactly one operator from experts_matmul_mul"
    );

    let cpu_num = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let started_at = Instant::now();
    start(operator_queue.take(), SEQUENCE_LENGTH, BATCH_SIZE);
    let elapsed = started_at.elapsed();

    let matmul_flops_per_step =
        2.0f64 * routed_pairs as f64 * INTERMEDIATE_SIZE as f64 * HIDDEN_SIZE as f64;
    let effective_gflops =
        (matmul_flops_per_step * SEQUENCE_LENGTH as f64) / elapsed.as_secs_f64() / 1e9;

    println!(
        "experts_matmul_mul perf: batch={}, experts={}, topk={}, inter={}, hidden={}, routed_pairs={}, threads={}, elapsed={:?}, effective_gflops={:.2}",
        BATCH_SIZE,
        NUM_EXPERTS,
        TOPK,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        routed_pairs,
        cpu_num,
        elapsed,
        effective_gflops
    );
}
