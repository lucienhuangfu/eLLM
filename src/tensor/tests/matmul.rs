use super::common::*;
use super::*;
// MatMul3 tests — weights now NT (N×K row-major)
// ============================================================

#[test]
fn test_matmul3_f16() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }
    init_f16_tensor_test_runtime();

    let batch_size = 3;
    let hidden_size = 64;
    let q_dim = 128;
    let kv_dim = 128;
    let head_dim = 128;

    let input_shape = vec![batch_size, hidden_size];
    let input_tensor =
        Tensor::<f16>::from_mem_pool(input_shape.clone(), "model.layers.0.input".to_string());

    // Tensor shape is [N, K] but raw mem_mgr should be NT: N×K
    let q_weight = Tensor::<f16>::from_mem_pool(vec![q_dim, hidden_size], "q.weight".to_string());
    let k_weight = Tensor::<f16>::from_mem_pool(vec![kv_dim, hidden_size], "k.weight".to_string());
    let v_weight = Tensor::<f16>::from_mem_pool(vec![kv_dim, hidden_size], "v.weight".to_string());
    let q_norm_weight = Tensor::<f16>::from_mem_pool(vec![head_dim], "q_norm.weight".to_string());
    let k_norm_weight = Tensor::<f16>::from_mem_pool(vec![head_dim], "k_norm.weight".to_string());

    let position_embedding =
        Tensor::<f16>::from_mem_pool(vec![head_dim], "rope.weight".to_string());

    // input
    let num_input = batch_size * hidden_size;
    let mut input_data = vec![0.0f16; num_input];
    for i in 0..batch_size {
        for k in 0..hidden_size {
            input_data[i * hidden_size + k] = (((i * 7 + k * 3) % 19) as f32 * 0.01) as f16;
        }
    }
    unsafe {
        input_tensor
            .data
            .copy_from_nonoverlapping(input_data.as_ptr(), num_input);
    }

    // weights: NT (N×K)
    let mut q_data_nt = vec![0.0f16; q_dim * hidden_size];
    let mut k_data_nt = vec![0.0f16; kv_dim * hidden_size];
    let mut v_data_nt = vec![0.0f16; kv_dim * hidden_size];

    for n in 0..q_dim {
        for k in 0..hidden_size {
            // 原来是 [k*q_dim + n]，现在改为 [n*hidden + k]
            q_data_nt[n * hidden_size + k] = (((k * 5 + n * 11) % 23) as f32 * 0.01) as f16;
        }
    }
    for n in 0..kv_dim {
        for k in 0..hidden_size {
            k_data_nt[n * hidden_size + k] = (((k * 3 + n * 7) % 29) as f32 * 0.01) as f16;
            v_data_nt[n * hidden_size + k] = (((k * 9 + n * 4) % 31) as f32 * 0.01) as f16;
        }
    }

    unsafe {
        q_weight
            .data
            .copy_from_nonoverlapping(q_data_nt.as_ptr(), q_data_nt.len());
        k_weight
            .data
            .copy_from_nonoverlapping(k_data_nt.as_ptr(), k_data_nt.len());
        v_weight
            .data
            .copy_from_nonoverlapping(v_data_nt.as_ptr(), v_data_nt.len());
        for i in 0..head_dim {
            *q_norm_weight.data.add(i) = 1.0f16;
            *k_norm_weight.data.add(i) = 1.0f16;
        }
    }

    let rope_data = rope_identity(head_dim);
    unsafe {
        position_embedding
            .data
            .copy_from_nonoverlapping(rope_data.as_ptr(), head_dim);
    }

    let params = MatMulParams {
        a_row_step_macro: 3,
        b_row_step_macro: 64,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let (q_out, k_out, v_out) = input_tensor.matmul3(
        &q_weight,
        &k_weight,
        &v_weight,
        &q_norm_weight,
        &k_norm_weight,
        &position_embedding,
        batch_size, // sequence_length
        1,          // batch_size
        1,          // kv_head_num
        1,          // group_num
        head_dim,
        true,
        params,
        "model.layers.0.matmul3".to_string(),
    );

    run_f16_queue(batch_size, 0, 1);

    let verify_matmul_nt = |output_tensor: &Tensor<f16>,
                            w_nt: &[f16],
                            n_dim: usize,
                            name: &str,
                            post_process_qk: bool| {
        let out_len = batch_size * n_dim;
        let out_data = unsafe { std::slice::from_raw_parts(output_tensor.data, out_len) };
        let mut expected = vec![0.0f32; out_len];

        for i in 0..batch_size {
            for j in 0..n_dim {
                let mut sum = 0.0f32;
                for k in 0..hidden_size {
                    let a_val = input_data[i * hidden_size + k] as f32;
                    let w_val = w_nt[j * hidden_size + k] as f32; // NT
                    sum += a_val * w_val;
                }
                expected[i * n_dim + j] = sum;
            }
        }

        if post_process_qk {
            apply_qk_post_process_ref_f16(&mut expected, batch_size, n_dim, head_dim, &rope_data);
        }

        for i in 0..batch_size {
            for j in 0..n_dim {
                let val = out_data[i * n_dim + j] as f32;
                assert!(
                    (val - expected[i * n_dim + j]).abs() < 0.5,
                    "{} mismatch at batch {}, col {}: got {}, expected {}",
                    name,
                    i,
                    j,
                    val,
                    expected[i * n_dim + j]
                );
            }
        }
    };

    verify_matmul_nt(&q_out, &q_data_nt, q_dim, "Q", true);
    verify_matmul_nt(&k_out, &k_data_nt, kv_dim, "K", true);
    verify_matmul_nt(&v_out, &v_data_nt, kv_dim, "V", false);
}

#[test]
fn test_tensor_matmul3_f16_seq1_batch24() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }
    init_f16_tensor_test_runtime();

    let batch_size = 24;
    let hidden_size = 64;

    let q_dim = 128;
    let kv_dim = 128;
    let head_dim = 128;

    let input_tensor = Tensor::<f16>::from_mem_pool(
        vec![batch_size, hidden_size],
        "model.layers.0.input".to_string(),
    );

    let q_weight = Tensor::<f16>::from_mem_pool(vec![q_dim, hidden_size], "q.weight".to_string());
    let k_weight = Tensor::<f16>::from_mem_pool(vec![kv_dim, hidden_size], "k.weight".to_string());
    let v_weight = Tensor::<f16>::from_mem_pool(vec![kv_dim, hidden_size], "v.weight".to_string());
    let q_norm_weight = Tensor::<f16>::from_mem_pool(vec![head_dim], "q_norm.weight".to_string());
    let k_norm_weight = Tensor::<f16>::from_mem_pool(vec![head_dim], "k_norm.weight".to_string());

    let position_embedding =
        Tensor::<f16>::from_mem_pool(vec![head_dim], "rope.weight".to_string());

    // input init
    let m = batch_size;
    let mut input_data = vec![0.0f16; m * hidden_size];
    for b in 0..batch_size {
        for kk in 0..hidden_size {
            input_data[b * hidden_size + kk] = (((b * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
        }
    }
    unsafe {
        input_tensor
            .data
            .copy_from_nonoverlapping(input_data.as_ptr(), input_data.len());
    }

    // weights init: NT (N×K)
    let mut q_data_nt = vec![0.0f16; q_dim * hidden_size];
    let mut k_data_nt = vec![0.0f16; kv_dim * hidden_size];
    let mut v_data_nt = vec![0.0f16; kv_dim * hidden_size];

    for n in 0..q_dim {
        for kk in 0..hidden_size {
            q_data_nt[n * hidden_size + kk] = (((kk * 5 + n * 11) % 23) as f32 * 0.01) as f16;
        }
    }
    for n in 0..kv_dim {
        for kk in 0..hidden_size {
            k_data_nt[n * hidden_size + kk] = (((kk * 3 + n * 7) % 29) as f32 * 0.01) as f16;
            v_data_nt[n * hidden_size + kk] = (((kk * 9 + n * 4) % 31) as f32 * 0.01) as f16;
        }
    }

    unsafe {
        q_weight
            .data
            .copy_from_nonoverlapping(q_data_nt.as_ptr(), q_data_nt.len());
        k_weight
            .data
            .copy_from_nonoverlapping(k_data_nt.as_ptr(), k_data_nt.len());
        v_weight
            .data
            .copy_from_nonoverlapping(v_data_nt.as_ptr(), v_data_nt.len());
        for i in 0..head_dim {
            *q_norm_weight.data.add(i) = 1.0f16;
            *k_norm_weight.data.add(i) = 1.0f16;
        }
    }

    let rope_data = rope_identity(head_dim);
    unsafe {
        position_embedding
            .data
            .copy_from_nonoverlapping(rope_data.as_ptr(), head_dim);
    }

    let params = MatMulParams {
        a_row_step_macro: 6,
        b_row_step_macro: 64,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let (q_out, k_out, v_out) = input_tensor.matmul3(
        &q_weight,
        &k_weight,
        &v_weight,
        &q_norm_weight,
        &k_norm_weight,
        &position_embedding,
        batch_size, // sequence_length
        1,          // batch_size
        1,          // kv_head_num
        1,          // group_num
        head_dim,
        true,
        params,
        "model.layers.0.matmul3".to_string(),
    );

    assert_eq!(q_out.shape, vec![batch_size, 1, q_dim]);
    assert_eq!(k_out.shape, vec![batch_size, 1, kv_dim]);
    assert_eq!(v_out.shape, vec![batch_size, 1, kv_dim]);
    let op = take_single_f16_operator(|op| matches!(op, Operator::MatMul3(_)));
    run_operator_all_threads(&op, batch_size, 0, 1);

    let q_len = m * q_dim;
    let kv_len = m * kv_dim;
    let q_got = unsafe { std::slice::from_raw_parts(q_out.data, q_len) };
    let k_got = unsafe { std::slice::from_raw_parts(k_out.data, kv_len) };
    let v_got = unsafe { std::slice::from_raw_parts(v_out.data, kv_len) };

    let check_nt = |got: &[f16], w_nt: &[f16], n_dim: usize, name: &str, post_process_qk: bool| {
        let mut expected = vec![0.0f32; m * n_dim];
        for i in 0..m {
            for j in 0..n_dim {
                let mut sum = 0.0f32;
                for kk in 0..hidden_size {
                    sum += (input_data[i * hidden_size + kk] as f32)
                        * (w_nt[j * hidden_size + kk] as f32);
                }
                expected[i * n_dim + j] = sum;
            }
        }

        if post_process_qk {
            apply_qk_post_process_ref_f16(&mut expected, m, n_dim, head_dim, &rope_data);
        }

        for i in 0..m {
            for j in 0..n_dim {
                let val = got[i * n_dim + j] as f32;
                assert!(
                    (val - expected[i * n_dim + j]).abs() < 0.5,
                    "{} mismatch at row {}, col {}: got {}, expected {}",
                    name,
                    i,
                    j,
                    val,
                    expected[i * n_dim + j]
                );
            }
        }
    };

    check_nt(q_got, &q_data_nt, q_dim, "Q", true);
    check_nt(k_got, &k_data_nt, kv_dim, "K", true);
    check_nt(v_got, &v_data_nt, kv_dim, "V", false);
}

// ============================================================
// MatMulTopK / matmul_local_topk — B is NT now
// ============================================================

#[test]
fn test_matmul_local_topk_f16() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }
    init_f16_tensor_test_runtime();

    let batch_size = 12;
    let hidden_size = 64; // K
    let intermediate_size = 96; // N
    let topk = 10;

    let input_tensor = Tensor::<f16>::from_mem_pool(
        vec![batch_size, hidden_size],
        "model.layers.0.input".to_string(),
    );

    let weight_tensor = Tensor::<f16>::from_mem_pool(
        vec![intermediate_size, hidden_size],
        "weight.weight".to_string(),
    );

    let m = batch_size;
    let k = hidden_size;
    let n = intermediate_size;

    // A
    let mut input_data = vec![0.0f16; m * k];
    for i in 0..m {
        for j in 0..k {
            input_data[i * k + j] = ((i + j) as f32 * 0.01) as f16;
        }
    }
    unsafe {
        input_tensor
            .data
            .copy_from_nonoverlapping(input_data.as_ptr(), m * k);
    }

    // B_nt: N×K
    let mut weight_data_nt = vec![0.0f16; n * k];
    for j in 0..n {
        for kk in 0..k {
            // 原来 (kk + j)*0.001 写在 KxN，现在写在 NT
            weight_data_nt[j * k + kk] = ((kk + j) as f32 * 0.001) as f16;
        }
    }
    unsafe {
        weight_tensor
            .data
            .copy_from_nonoverlapping(weight_data_nt.as_ptr(), n * k);
    }

    let params = MatMulParams {
        a_row_step_macro: 6,
        b_row_step_macro: 64,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let (indice_ptr, value_tensor) = input_tensor.matmul_local_topk(
        &weight_tensor,
        params,
        topk,
        avail_threads(),
        "model.layers.0.matmul_local_topk".to_string(),
    );

    let op = take_single_f16_operator(|op| matches!(op, Operator::MatMulTopK(_)));
    let buffer_thread_num = match &op {
        Operator::MatMulTopK(operator) => operator.thread_max(),
        _ => unreachable!(),
    };
    let run_thread_num = avail_threads().min(buffer_thread_num).max(1);
    run_operator_all_threads(&op, m, 0, run_thread_num);

    let out_len = m * buffer_thread_num * topk;
    let indices = unsafe { std::slice::from_raw_parts(indice_ptr, out_len) };
    let values = unsafe { std::slice::from_raw_parts(value_tensor.data, out_len) };

    for i in 0..m {
        // full reference C row (f32)
        let mut row_c = vec![0.0f32; n];
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += (input_data[i * k + kk] as f32) * (weight_data_nt[j * k + kk] as f32);
            }
            row_c[j] = sum;
        }

        let mut indexed_row: Vec<(usize, f32)> = row_c.into_iter().enumerate().collect();
        indexed_row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let expected_topk = &indexed_row[0..topk];

        let mut merged: Vec<(usize, f32)> = Vec::new();
        for tid in 0..run_thread_num {
            let offset = i * (buffer_thread_num * topk) + tid * topk;
            for r in 0..topk {
                merged.push((indices[offset + r], values[offset + r] as f32));
            }
        }
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let final_topk = &merged[0..topk];

        for r in 0..topk {
            let (exp_idx, exp_val) = expected_topk[r];
            let (got_idx, got_val) = final_topk[r];
            assert!(
                (got_val - exp_val).abs() < 0.05,
                "Value mismatch row {} rank {}: got {} exp {}",
                i,
                r,
                got_val,
                exp_val
            );
            if (got_val - exp_val).abs() < 1e-4 {
                assert_eq!(got_idx, exp_idx, "Index mismatch row {} rank {}", i, r);
            }
        }
    }
}

#[test]
fn test_matmul_local_topk_f16_no_ties_stable() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }
    init_f16_tensor_test_runtime();

    let batch_size = 12;
    let k = 64;
    let n = 128;
    let topk = 10;

    let input_tensor =
        Tensor::<f16>::from_mem_pool(vec![batch_size, k], "model.layers.0.input".to_string());

    let weight_tensor = Tensor::<f16>::from_mem_pool(vec![n, k], "weight.weight".to_string());

    // A = 1
    let m = batch_size;
    let mut a = vec![0.0f16; m * k];
    for x in &mut a {
        *x = 1.0f16;
    }
    unsafe {
        input_tensor
            .data
            .copy_from_nonoverlapping(a.as_ptr(), a.len());
    }

    // B_nt: bias(j) = f16(j*0.001), no ties
    let mut b_nt = vec![0.0f16; n * k];
    for j in 0..n {
        let bias_f16: f16 = (j as f32 * 0.001) as f16;
        for kk in 0..k {
            b_nt[j * k + kk] = bias_f16;
        }
    }
    unsafe {
        weight_tensor
            .data
            .copy_from_nonoverlapping(b_nt.as_ptr(), b_nt.len());
    }

    let params = MatMulParams {
        a_row_step_macro: 6,
        b_row_step_macro: 64,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let (indice_ptr, value_tensor) = input_tensor.matmul_local_topk(
        &weight_tensor,
        params,
        topk,
        avail_threads(),
        "model.layers.0.matmul_local_topk".to_string(),
    );

    let op = take_single_f16_operator(|op| matches!(op, Operator::MatMulTopK(_)));
    let buffer_thread_num = match &op {
        Operator::MatMulTopK(operator) => operator.thread_max(),
        _ => unreachable!(),
    };
    let run_thread_num = avail_threads().min(buffer_thread_num).max(1);
    run_operator_all_threads(&op, m, 0, run_thread_num);

    let out_len = m * buffer_thread_num * topk;
    let indices = unsafe { std::slice::from_raw_parts(indice_ptr, out_len) };
    let values = unsafe { std::slice::from_raw_parts(value_tensor.data, out_len) };

    let expected_indices: Vec<usize> = (0..topk).map(|r| n - 1 - r).collect();

    let expected_value = |j: usize| -> f32 {
        let bias_f16: f16 = (j as f32 * 0.001) as f16;
        (k as f32) * (bias_f16 as f32)
    };

    for row in 0..m {
        let mut merged: Vec<(usize, f32)> = Vec::with_capacity(run_thread_num * topk);
        for tid in 0..run_thread_num {
            let base = row * (buffer_thread_num * topk) + tid * topk;
            for r in 0..topk {
                merged.push((indices[base + r], values[base + r] as f32));
            }
        }

        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let final_topk = &merged[..topk];

        for r in 0..topk {
            let (got_idx, got_val) = final_topk[r];
            let exp_idx = expected_indices[r];
            let exp_val = expected_value(exp_idx);

            assert_eq!(
                got_idx, exp_idx,
                "Index mismatch at row {}, rank {}: got {}, expected {}",
                row, r, got_idx, exp_idx
            );

            assert!(
                (got_val - exp_val).abs() < 0.1,
                "Value mismatch at row {}, rank {}: got {}, expected {}",
                row,
                r,
                got_val,
                exp_val
            );
        }
    }
}

// ============================================================
// MatMul / MatMulAdd — B is NT now
// ============================================================

#[test]
fn test_matmul_f16() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }
    init_f16_tensor_test_runtime();

    let batch_size = 12;
    let hidden_size = 64; // K
    let intermediate_size = 96; // N

    let input_tensor = Tensor::<f16>::from_mem_pool(
        vec![batch_size, hidden_size],
        "model.layers.0.input".to_string(),
    );

    let weight_tensor = Tensor::<f16>::from_mem_pool(
        vec![intermediate_size, hidden_size],
        "weight.weight".to_string(),
    );

    let m = batch_size;
    let k = hidden_size;
    let n = intermediate_size;

    let mut input_data = vec![0.0f16; m * k];
    for i in 0..m {
        for j in 0..k {
            input_data[i * k + j] = (((i * 7 + j * 3) % 19) as f32 * 0.1) as f16;
        }
    }
    unsafe {
        input_tensor
            .data
            .copy_from_nonoverlapping(input_data.as_ptr(), m * k);
    }

    // B_nt: N×K
    let mut weight_data_nt = vec![0.0f16; n * k];
    for j in 0..n {
        for kk in 0..k {
            weight_data_nt[j * k + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.1) as f16;
        }
    }
    unsafe {
        weight_tensor
            .data
            .copy_from_nonoverlapping(weight_data_nt.as_ptr(), n * k);
    }

    let params = MatMulParams {
        a_row_step_macro: 6,
        b_row_step_macro: 64,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let output_tensor = input_tensor.matmul(
        &weight_tensor,
        params,
        1,
        false,
        "model.layers.0.matmul".to_string(),
    );

    let op = take_single_f16_operator(|op| matches!(op, Operator::MatMul(_)));
    run_operator_all_threads(&op, m, 0, 1);

    let out_len = m * n;
    let output_data = unsafe { std::slice::from_raw_parts(output_tensor.data, out_len) };

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let a = input_data[i * k + kk] as f32;
                let b = weight_data_nt[j * k + kk] as f32; // NT
                sum += a * b;
            }
            let val = output_data[i * n + j] as f32;
            assert!(
                (val - sum).abs() < 0.5,
                "Mismatch at batch {}, col {}: got {}, expected {}",
                i,
                j,
                val,
                sum
            );
        }
    }
}

#[test]
fn test_matmul_add_f16() {
    if !std::arch::is_x86_feature_detected!("avx512fp16") {
        println!("AVX512FP16 not supported, skipping test.");
        return;
    }
    init_f16_tensor_test_runtime();

    let batch_size = 12;
    let hidden_size = 64; // K
    let intermediate_size = 96; // N

    let input_tensor = Tensor::<f16>::from_mem_pool(
        vec![batch_size, hidden_size],
        "model.layers.0.input".to_string(),
    );

    let weight_tensor = Tensor::<f16>::from_mem_pool(
        vec![intermediate_size, hidden_size],
        "weight.weight".to_string(),
    );

    let bias_tensor = Tensor::<f16>::from_mem_pool(
        vec![batch_size, intermediate_size],
        "bias.weight".to_string(),
    );

    let m = batch_size;
    let k = hidden_size;
    let n = intermediate_size;

    // A
    let mut input_data = vec![0.0f16; m * k];
    for i in 0..m {
        for j in 0..k {
            input_data[i * k + j] = (((i * 7 + j * 3) % 19) as f32 * 0.1) as f16;
        }
    }
    unsafe {
        input_tensor
            .data
            .copy_from_nonoverlapping(input_data.as_ptr(), m * k);
    }

    // B_nt
    let mut weight_data_nt = vec![0.0f16; n * k];
    for j in 0..n {
        for kk in 0..k {
            weight_data_nt[j * k + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.1) as f16;
        }
    }
    unsafe {
        weight_tensor
            .data
            .copy_from_nonoverlapping(weight_data_nt.as_ptr(), n * k);
    }

    // bias
    let mut bias_data = vec![0.0f16; m * n];
    for i in 0..m {
        for j in 0..n {
            bias_data[i * n + j] = (((i * 2 + j * 5) % 17) as f32 * 0.1) as f16;
        }
    }
    unsafe {
        bias_tensor
            .data
            .copy_from_nonoverlapping(bias_data.as_ptr(), m * n);
    }

    let params = MatMulParams {
        a_row_step_macro: 6,
        b_row_step_macro: 64,
        column_step_macro: 64,
        a_row_step_micro: 3,
        b_row_step_micro: 32,
    };

    let output_tensor = input_tensor.matmul_add(
        &weight_tensor,
        &bias_tensor,
        params,
        false,
        "model.layers.0.matmul_add".to_string(),
    );

    let op = take_single_f16_operator(|op| matches!(op, Operator::MatMulAdd(_)));
    run_operator_all_threads(&op, m, 0, 1);

    let out_len = m * n;
    let output_data = unsafe { std::slice::from_raw_parts(output_tensor.data, out_len) };

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += (input_data[i * k + kk] as f32) * (weight_data_nt[j * k + kk] as f32);
            }
            sum += bias_data[i * n + j] as f32;

            let val = output_data[i * n + j] as f32;
            assert!(
                (val - sum).abs() < 0.5,
                "Mismatch at batch {}, col {}: got {}, expected {}",
                i,
                j,
                val,
                sum
            );
        }
    }
}

// ============================================================
