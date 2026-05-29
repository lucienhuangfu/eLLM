use super::common::*;
use super::*;

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

            op.run(batch, 0, cpu_num, thread_id, &[], &[], &mut Vec::new());
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[inline]
fn skip_if_no_avx512fp16() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported on this machine, skipping test.");
    }
}

#[test]
fn test_matmul_new_uses_bnt_directly_f16() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }

    const K: usize = 8;
    const N: usize = 6;
    const M: usize = 3;

    // ✅ NEW contract: B is already NT (N×K) row-major
    let mut a = vec![0.0f16; M * K];
    let mut b_nt = vec![0.0f16; N * K];
    let mut c = vec![0.0f16; M * N];

    // b_nt 用唯一值：b_nt[j*K + kk] = 100*j + kk
    for j in 0..N {
        for kk in 0..K {
            let v = (100 * j + kk) as f32;
            b_nt[j * K + kk] = v as f16;
        }
    }

    let params = MatMulParams {
        a_row_step_macro: 3,
        b_row_step_macro: 32,
        column_step_macro: 4, // kc
        a_row_step_micro: 3,
        b_row_step_micro: 2, // nr
    };

    let matmul = unsafe {
        MatMul::<f16>::new(
            a.as_ptr(),
            b_nt.as_ptr(),
            c.as_mut_ptr(),
            false,
            params,
            M,
            N,
            K,
            false,
        )
    };

    // 只验证：内部使用的 B 视图（ptr2.ptr）在“语义上”保持 b_nt 的 N×K 索引一致
    // 注意：这里不要求 ptr2.ptr 与输入同一地址（实现可能做 pack/copy）
    let internal_b_nt = unsafe { std::slice::from_raw_parts(matmul.ptr2.ptr, N * K) };
    for j in 0..N {
        for kk in 0..K {
            let got = internal_b_nt[j * K + kk] as f32;
            let expected = b_nt[j * K + kk] as f32;
            assert_abs_diff_eq!(got, expected, epsilon = 0.0);
        }
    }
}

#[test]
fn test_matmul_panel_threads_available_f16() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }

    const M: usize = 3;
    const K: usize = 64;
    const N: usize = 32;

    let a = vec![0.0f16; M * K];
    let b_nt = vec![0.0f16; N * K];
    let mut c = vec![0.0f16; M * N];

    let params = MatMulParams {
        a_row_step_macro: 3,
        b_row_step_macro: 32,
        column_step_macro: 64, // kc
        a_row_step_micro: 3,
        b_row_step_micro: 32, // nr
    };

    let matmul = unsafe {
        MatMul::<f16>::new(
            a.as_ptr(),
            b_nt.as_ptr(),
            c.as_mut_ptr(),
            false,
            params,
            M,
            N,
            K,
            false,
        )
    };

    assert!(matmul.panel_threads() >= 1);
}

#[test]
fn test_matmul_add_new_uses_bnt_directly_f16() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }

    const K: usize = 8;
    const N: usize = 6;
    const M: usize = 3;

    let a = vec![0.0f16; M * K];
    let mut b_nt = vec![0.0f16; N * K];
    let residual = vec![0.0f16; M * N];
    let mut c = vec![0.0f16; M * N];

    for j in 0..N {
        for kk in 0..K {
            let v = (100 * j + kk) as f32;
            b_nt[j * K + kk] = v as f16;
        }
    }

    let params = MatMulParams {
        a_row_step_macro: 3,
        b_row_step_macro: 32,
        column_step_macro: 4,
        a_row_step_micro: 3,
        b_row_step_micro: 2,
    };

    let matmul_add = unsafe {
        MatMulAdd::<f16>::new(
            a.as_ptr(),
            b_nt.as_ptr(),
            residual.as_ptr(),
            c.as_mut_ptr(),
            params,
            M,
            N,
            K,
            false,
        )
    };

    let internal_b_nt = unsafe { std::slice::from_raw_parts(matmul_add.ptr2.ptr, N * K) };
    for j in 0..N {
        for kk in 0..K {
            let got = internal_b_nt[j * K + kk] as f32;
            let expected = b_nt[j * K + kk] as f32;
            assert_abs_diff_eq!(got, expected, epsilon = 0.0);
        }
    }
}

#[test]
fn test_matmul_add_panel_threads_available_f16() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }

    const M: usize = 3;
    const K: usize = 64;
    const N: usize = 32;

    let a = vec![0.0f16; M * K];
    let b_nt = vec![0.0f16; N * K];
    let residual = vec![0.0f16; M * N];
    let mut c = vec![0.0f16; M * N];

    let params = MatMulParams {
        a_row_step_macro: 3,
        b_row_step_macro: 32,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let matmul_add = unsafe {
        MatMulAdd::<f16>::new(
            a.as_ptr(),
            b_nt.as_ptr(),
            residual.as_ptr(),
            c.as_mut_ptr(),
            params,
            M,
            N,
            K,
            false,
        )
    };

    assert!(matmul_add.panel_threads() >= 1);
}

#[test]
fn test_matmul_runner_f16_multi_tile_and_threads() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }

    const M: usize = 12;
    const K: usize = 64;
    const N: usize = 128;

    let thread_num = avail_threads().min(16);

    let mut a = vec![0.0f16; M * K];
    // ✅ NEW: B is already NT: N×K row-major
    let mut b_nt = vec![0.0f16; N * K];
    let mut c = vec![0.0f16; M * N];

    for i in 0..M {
        for kk in 0..K {
            a[i * K + kk] = (((i * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
        }
    }
    for j in 0..N {
        for kk in 0..K {
            // 同 pattern，只是按 NT 写
            b_nt[j * K + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.01) as f16;
        }
    }

    let params = MatMulParams {
        a_row_step_macro: 6,
        b_row_step_macro: 128,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let matmul = unsafe {
        MatMul::<f16>::new(
            a.as_ptr(),
            b_nt.as_ptr(),
            c.as_mut_ptr(),
            false,
            params,
            M,
            N,
            K,
            false,
        )
    };

    let cpu_num = thread_num.min(matmul.panel_threads()).max(1);

    for tid in 0..cpu_num {
        matmul.run(M, 0, cpu_num, tid);
    }

    // reference: sum += A[i,kk] * B_nt[j,kk]
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0f32;
            for kk in 0..K {
                sum += (a[i * K + kk] as f32) * (b_nt[j * K + kk] as f32);
            }
            let got = c[i * N + j] as f32;
            assert_abs_diff_eq!(got, sum, epsilon = 5e-1);
        }
    }
}

#[test]
#[ignore = "performance test"]
fn test_tensor_matmul_perf_single_operator_parallel() {
    if !cfg!(target_arch = "x86_64") || !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }

    const SEQUENCE_CHUNK_SIZE: usize = 1;
    const BATCH_SIZE: usize = 384;
    const K: usize = 1536;
    const N: usize = 1536;

    let input_tensor = Tensor::<f16>::from_mem_pool(
        vec![SEQUENCE_CHUNK_SIZE, BATCH_SIZE, K],
        "perf.matmul.input".to_string(),
    );

    let weight_tensor = Tensor::<f16>::from_mem_pool(vec![N, K], "perf.matmul.weight".to_string());

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
        false,
        "perf.matmul".to_string(),
    );

    let queue = f16::take_operator_queue();
    assert_eq!(
        queue.len(),
        1,
        "expected exactly one operator from tensor.matmul"
    );
    assert!(matches!(&queue[0], Operator::MatMul(_)));
    let operator = Arc::new(queue[0].clone());

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
