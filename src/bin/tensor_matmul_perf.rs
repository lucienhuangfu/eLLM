#![feature(f16)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::f16;
use std::rc::Rc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use ellm::compiler::operator::Operator;
use ellm::init::matmul_params::MatMulParams;
use ellm::memory::cache::Cache;
use ellm::ptensor::tensor::Tensor;

const SEQUENCE_CHUNK_SIZE: usize = 1;
const BATCH_SIZE: usize = 384;
const K: usize = 1536;
const N: usize = 1536;

fn avail_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn run_operator_parallel_once(op: Arc<Operator<f16>>, batch: usize, cpu_num: usize) {
    let core_ids = core_affinity::get_core_ids();
    let mut handles = Vec::with_capacity(cpu_num);

    for thread_id in 0..cpu_num {
        let op = Arc::clone(&op);
        let core_id = core_ids
            .as_ref()
            .and_then(|ids| ids.get(thread_id % ids.len()).copied());

        let handle = thread::spawn(move || {
            if let Some(core_id) = core_id {
                core_affinity::set_for_current(core_id);
            }

            op.run(0, 1, batch, cpu_num, thread_id);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

fn main() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported on this machine, skipping tensor.matmul perf run.");
        return;
    }

    println!(
        "tensor.matmul perf init: seq_chunk={}, batch={}, n={}, k={}",
        SEQUENCE_CHUNK_SIZE, BATCH_SIZE, N, K
    );

    let cache: Rc<RefCell<Cache<f16>>> = Rc::new(RefCell::new(Cache::new(HashMap::new())));
    let operator_queue: Rc<RefCell<Vec<Operator<f16>>>> = Rc::new(RefCell::new(Vec::new()));

    let input_tensor = Tensor::<f16>::from_cache(
        vec![SEQUENCE_CHUNK_SIZE, BATCH_SIZE, K],
        "perf.matmul.input".to_string(),
        cache.clone(),
        operator_queue.clone(),
    );

    let weight_tensor = Tensor::<f16>::from_cache(
        vec![N, K],
        "perf.matmul.weight".to_string(),
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
        a_row_step_macro: 96,
        b_row_step_macro: 128,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let _output_tensor = input_tensor.matmul(
        &weight_tensor,
        params,
        SEQUENCE_CHUNK_SIZE,
        "perf.matmul".to_string(),
    );

    let operator = {
        let queue = operator_queue.borrow();
        assert_eq!(
            queue.len(),
            1,
            "expected exactly one operator from tensor.matmul"
        );
        Arc::new(queue[0].clone())
    };

    let panel_threads = match operator.as_ref() {
        Operator::MatMul(runner) => runner.panel_threads(),
        _ => panic!("tensor.matmul should enqueue a MatMul operator"),
    };

    let cpu_num = avail_threads().min(panel_threads.max(1)).min(16);
    let start = Instant::now();
    run_operator_parallel_once(operator, BATCH_SIZE, cpu_num);
    let elapsed = start.elapsed();

    let flops = 2.0f64 * BATCH_SIZE as f64 * N as f64 * K as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    println!(
        "tensor.matmul perf: batch={}, n={}, k={}, threads={}, elapsed={:?}, gflops={:.2}",
        BATCH_SIZE, N, K, cpu_num, elapsed, gflops
    );
}
