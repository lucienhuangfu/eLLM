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
const HIDDEN_SIZE: usize = 2048;
const INTERMEDIATE_SIZE: usize = 4096;

fn main() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!(
            "AVX512FP16 not supported on this machine, skipping experts_matmul_silu_mul_matmul perf run."
        );
        return;
    }

    println!(
        "experts_matmul_silu_mul_matmul perf init: seq_len={}, seq_chunk={}, batch={}, experts={}, topk={}, hidden={}, inter={}",
        SEQUENCE_LENGTH,
        SEQUENCE_CHUNK_SIZE,
        BATCH_SIZE,
        NUM_EXPERTS,
        TOPK,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE
    );

    let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
    let operator_queue = Rc::new(RefCell::new(Vec::new()));

    let input_tensor = Tensor::<f16>::from_cache(
        vec![SEQUENCE_CHUNK_SIZE, BATCH_SIZE, HIDDEN_SIZE],
        "model.layers.0.perf.experts_silu.input".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let gate_weight = Tensor::<f16>::from_cache(
        vec![NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE],
        "model.layers.0.perf.experts_silu.gate_weight".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let up_weight = Tensor::<f16>::from_cache(
        vec![NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE],
        "model.layers.0.perf.experts_silu.up_weight".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let num_tokens = SEQUENCE_CHUNK_SIZE * BATCH_SIZE;
    let experts_indicator = allocate_init(NUM_EXPERTS, false);
    let indice_ptr = allocate_init(NUM_EXPERTS * num_tokens, false);

    let mut routed_pairs = 0usize;
    unsafe {
        for t in 0..num_tokens {
            for slot in 0..TOPK {
                let expert = (t * TOPK + slot) % NUM_EXPERTS;
                *experts_indicator.add(expert) = true;
                *indice_ptr.add(expert * num_tokens + t) = true;
                routed_pairs += 1;
            }
        }
    }

    for b in 0..BATCH_SIZE {
        for kk in 0..HIDDEN_SIZE {
            let value = (((b * 17 + kk * 13) % 97) as f32 * 0.01) as f16;
            unsafe {
                *input_tensor.data.add(b * HIDDEN_SIZE + kk) = value;
            }
        }
    }

    let weight_elems_per_expert = INTERMEDIATE_SIZE * HIDDEN_SIZE;
    for e in 0..NUM_EXPERTS {
        let gate_base = e * weight_elems_per_expert;
        let up_base = e * weight_elems_per_expert;
        for ii in 0..INTERMEDIATE_SIZE {
            let row_base = ii * HIDDEN_SIZE;
            for kk in 0..HIDDEN_SIZE {
                let gate_value = (((e * 19 + ii * 7 + kk * 5) % 89) as f32 * 0.01) as f16;
                let up_value = (((e * 23 + ii * 11 + kk * 3) % 83) as f32 * 0.01) as f16;
                unsafe {
                    *gate_weight.data.add(gate_base + row_base + kk) = gate_value;
                    *up_weight.data.add(up_base + row_base + kk) = up_value;
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

    let _output = input_tensor.experts_matmul_silu_mul_matmul(
        &gate_weight,
        &up_weight,
        experts_indicator,
        indice_ptr,
        params,
        "model.layers.0.perf.experts_silu".to_string(),
    );

    assert_eq!(
        operator_queue.borrow().len(),
        1,
        "expected exactly one operator from experts_matmul_silu_mul_matmul"
    );

    let cpu_num = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let started_at = Instant::now();
    start(operator_queue.take(), SEQUENCE_LENGTH, BATCH_SIZE);
    let elapsed = started_at.elapsed();

    let matmul_flops_per_step =
        4.0f64 * routed_pairs as f64 * HIDDEN_SIZE as f64 * INTERMEDIATE_SIZE as f64;
    let effective_gflops =
        (matmul_flops_per_step * SEQUENCE_LENGTH as f64) / elapsed.as_secs_f64() / 1e9;

    println!(
        "experts_matmul_silu_mul_matmul perf: batch={}, experts={}, topk={}, hidden={}, inter={}, routed_pairs={}, threads={}, elapsed={:?}, effective_gflops={:.2}",
        BATCH_SIZE,
        NUM_EXPERTS,
        TOPK,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        routed_pairs,
        cpu_num,
        elapsed,
        effective_gflops
    );
}
