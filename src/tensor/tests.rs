use crate::kernel::common::matmul_params::MatMulParams;
use crate::mem_mgr::allocator::AlignedBox;
use crate::operators::linear::{MatMul, MatMulAdd};
use crate::operators::operator::Operator;
use crate::runtime::sequence_slice::SequenceSlice;
use crate::runtime::{Phase, SequenceState};

use super::Tensor;

#[cfg(test)]
mod test {
    use super::*;
    use crate::mem_mgr::mem_pool::GlobalMemPool;
    use crate::operators::expert::expert_routing::routing_from_dense;
    use crate::tensor::GlobalOperatorQueue;
    use approx::{assert_abs_diff_eq, assert_ulps_eq};
    use std::collections::HashMap;
    use std::f16;
    use std::mem;

    // ============================================================
    // helpers
    // ============================================================

    fn avail_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    fn init_f16_tensor_test_runtime() {
        f16::init_global(HashMap::new());
        f16::init_operator_queue();
    }

    #[inline]
    fn f32_from_f16(x: f16) -> f32 {
        // bitcast based f16->f32 (与你原实现一致)
        let bits: u16 = unsafe { mem::transmute(x) };
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits & 0x7C00) >> 10;
        let mant = bits & 0x03FF;

        let f_bits: u32 = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                let mut m = mant as u32;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let exp_f = (e + 127) as u32;
                sign | (exp_f << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            let exp_f = 0xFFu32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        } else {
            let exp_f = (exp as i32 - 15 + 127) as u32;
            sign | (exp_f << 23) | ((mant as u32) << 13)
        };

        f32::from_bits(f_bits)
    }

    fn run_operator_all_threads(
        op: &Operator<f16>,
        prefill_size: usize,
        decode_size: usize,
        cpu_num: usize,
    ) {
        for tid in 0..cpu_num {
            op.run(
                prefill_size,
                decode_size,
                cpu_num,
                tid,
                &[],
                &[],
                &mut Vec::new(),
            );
        }
    }

    fn take_single_f16_operator<F>(matches_expected: F) -> Operator<f16>
    where
        F: FnOnce(&Operator<f16>) -> bool,
    {
        let queue = f16::take_operator_queue();
        assert_eq!(queue.len(), 1);
        assert!(matches_expected(&queue[0]));
        queue.into_iter().next().unwrap()
    }

    fn run_f16_queue(prefill_size: usize, decode_size: usize, cpu_num: usize) {
        let queue = f16::take_operator_queue();
        assert_eq!(queue.len(), 1);
        for op in queue.iter() {
            run_operator_all_threads(op, prefill_size, decode_size, cpu_num);
        }
    }

    fn rope_identity(head_dim: usize) -> Vec<f16> {
        let mut rope = vec![0.0f16; head_dim];
        for i in (0..head_dim).step_by(2) {
            rope[i] = 1.0f16;
        }
        rope
    }

    fn rms_norm_f32_in_place(row: &mut [f32]) {
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let rrms = 1.0f32 / (sum_sq / row.len() as f32 + 1e-6).sqrt();
        for v in row {
            *v *= rrms;
        }
    }

    fn apply_qk_post_process_ref_f16(
        data: &mut [f32],
        rows: usize,
        cols: usize,
        head_dim: usize,
        rope: &[f16],
    ) {
        for row in 0..rows {
            for head_base in (0..cols).step_by(head_dim) {
                let head = &mut data[row * cols + head_base..row * cols + head_base + head_dim];
                rms_norm_f32_in_place(head);
                for i in (0..head_dim).step_by(2) {
                    let a = head[i];
                    let b = head[i + 1];
                    let c = rope[i] as f32;
                    let d = rope[i + 1] as f32;
                    head[i] = a * c - b * d;
                    head[i + 1] = a * d + b * c;
                }
            }
        }
    }

    #[inline]
    fn silu_f32(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // NT layout helpers:
    // B_nt is N×K row-major => b_nt[j*K + kk]
    #[inline]
    fn idx_b_nt(j: usize, kk: usize, k: usize) -> usize {
        j * k + kk
    }

    // ============================================================
    // TopKSoftmax tests (unchanged - not matmul RHS related)
    // ============================================================

    #[test]
    fn test_topk_softmax_f32() {
        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        let batch_size = 2;
        let num_topk = 8;
        let thread_num = 2;
        let num_candidates_per_thread = num_topk;
        let num_candidates = num_candidates_per_thread * thread_num;
        let eos_id = 100;
        let mut batch_temperature = vec![1.0f32; batch_size];

        let value_shape = vec![batch_size, num_candidates];
        let value_tensor =
            Tensor::<f32>::from_mem_pool(value_shape, "model.layers.0.values".to_string());

        let sums_shape = vec![batch_size, thread_num];
        let sums_tensor =
            Tensor::<f32>::from_mem_pool(sums_shape, "model.layers.0.sums".to_string());

        let mut output_sequences = vec![0usize; batch_size * 2];

        let values0: Vec<f32> = (0..num_candidates).map(|i| 5.0 - i as f32 * 0.1).collect();
        let indices0: Vec<usize> = (0..num_candidates).collect();

        let values1: Vec<f32> = (0..num_candidates).map(|i| 8.0 - i as f32 * 0.2).collect();
        let indices1: Vec<usize> = (100..(100 + num_candidates)).collect();

        let mut all_values = Vec::new();
        all_values.extend_from_slice(&values0);
        all_values.extend_from_slice(&values1);

        let mut all_indices = Vec::new();
        all_indices.extend_from_slice(&indices0);
        all_indices.extend_from_slice(&indices1);

        let indices_ptr = all_indices.as_ptr();

        let mut batch_list = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            batch_list.push(SequenceState {
                filling_length: 0,
                sequence_index: 0,
                kv_index: 0,
                phase: Phase::Decode,
                // prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            });
        }
        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut decode_lists = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for batch_index in start..end {
                slices.push(SequenceSlice {
                    batch_index,
                    sequence_index: 0,
                    token_start_index: batch_index,
                    length: 1,
                    last_token_flag: true,
                });
            }
            decode_lists.push(slices);
        }
        let decode_list = decode_lists.iter().flatten().cloned().collect::<Vec<_>>();

        unsafe {
            value_tensor
                .data
                .copy_from_nonoverlapping(all_values.as_ptr(), all_values.len());
            let _ = sums_tensor; // sums currently unused
        }

        let (output_indices_ptr, output_value_tensor) = value_tensor.topk_softmax(
            indices_ptr,
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            1,
            num_topk,
            1.0f32,
            0.0f32,
            false,
            vec![eos_id],
            "model.layers.0.topk_softmax".to_string(),
        );

        let operator_queue = f32::take_operator_queue();
        for i in 0..thread_num {
            for op in operator_queue.iter() {
                if let Operator::TopKSoftmax(operator) = op {
                    operator.run(
                        batch_size,
                        1,
                        thread_num,
                        i,
                        &[],
                        &decode_list,
                        &mut batch_list,
                    );
                } else {
                    op.run(batch_size, 1, thread_num, i, &[], &[], &mut Vec::new());
                }
            }
        }

        let num_tokens = batch_size;
        let output_indices =
            unsafe { std::slice::from_raw_parts(output_indices_ptr, num_tokens * num_topk) };
        let output_values =
            unsafe { std::slice::from_raw_parts(output_value_tensor.data, num_tokens * num_topk) };

        // token 0
        let mut candidates0: Vec<_> = indices0.iter().zip(values0.iter()).collect();
        candidates0.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values0: Vec<f32> = candidates0.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val0 = top_values0[0];
        let exps0: Vec<f32> = top_values0.iter().map(|v| (v - max_val0).exp()).collect();
        let sum_exps0: f32 = exps0.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[i], *candidates0[i].0);
            assert_ulps_eq!(output_values[i], exps0[i] / sum_exps0, max_ulps = 4);
        }

        // token 1
        let mut candidates1: Vec<_> = indices1.iter().zip(values1.iter()).collect();
        candidates1.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values1: Vec<f32> = candidates1.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val1 = top_values1[0];
        let exps1: Vec<f32> = top_values1.iter().map(|v| (v - max_val1).exp()).collect();
        let sum_exps1: f32 = exps1.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[num_topk + i], *candidates1[i].0);
            assert_ulps_eq!(
                output_values[num_topk + i],
                exps1[i] / sum_exps1,
                max_ulps = 4
            );
        }

        assert_eq!(output_sequences[0], *candidates0[0].0);
        assert_eq!(output_sequences[1], *candidates1[0].0);
    }

    #[test]
    fn test_topk_softmax_f16() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }

        f16::init_global(HashMap::new());
        f16::init_operator_queue();

        let batch_size = 2;
        let num_topk = 8;
        let thread_num = 2;
        let num_candidates_per_thread = num_topk;
        let num_candidates = num_candidates_per_thread * thread_num;
        let eos_id = 100;
        let mut batch_temperature = vec![1.0f16; batch_size];

        let value_shape = vec![batch_size, num_candidates];
        let value_tensor =
            Tensor::<f16>::from_mem_pool(value_shape, "model.layers.0.values".to_string());

        let sums_shape = vec![batch_size, thread_num];
        let sums_tensor =
            Tensor::<f16>::from_mem_pool(sums_shape, "model.layers.0.sums".to_string());

        let mut output_sequences = vec![0usize; batch_size];

        let values0: Vec<f32> = (0..num_candidates).map(|i| 5.0 - i as f32 * 0.1).collect();
        let indices0: Vec<usize> = (0..num_candidates).collect();

        let values1: Vec<f32> = (0..num_candidates).map(|i| 8.0 - i as f32 * 0.2).collect();
        let indices1: Vec<usize> = (100..(100 + num_candidates)).collect();

        let mut all_values_f32 = Vec::new();
        all_values_f32.extend_from_slice(&values0);
        all_values_f32.extend_from_slice(&values1);
        let all_values: Vec<f16> = all_values_f32.iter().map(|&x| x as f16).collect();

        let mut all_indices = Vec::new();
        all_indices.extend_from_slice(&indices0);
        all_indices.extend_from_slice(&indices1);

        let indices_ptr = all_indices.as_ptr();

        let mut batch_list = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            batch_list.push(SequenceState {
                filling_length: 0,
                sequence_index: 0,
                kv_index: 0,
                phase: Phase::Decode,
                // prompt_length: i,
                notify: std::sync::Arc::new(tokio::sync::Notify::new()),
            });
        }
        let tokens_per_thread = (batch_size + thread_num - 1) / thread_num;
        let mut decode_lists = Vec::with_capacity(thread_num);
        for tid in 0..thread_num {
            let start = tid * tokens_per_thread;
            let end = (start + tokens_per_thread).min(batch_size);
            let mut slices = Vec::with_capacity(end.saturating_sub(start));
            for batch_index in start..end {
                slices.push(SequenceSlice {
                    batch_index,
                    sequence_index: 0,
                    token_start_index: batch_index,
                    length: 1,
                    last_token_flag: true,
                });
            }
            decode_lists.push(slices);
        }
        let decode_list = decode_lists.iter().flatten().cloned().collect::<Vec<_>>();

        unsafe {
            value_tensor
                .data
                .copy_from_nonoverlapping(all_values.as_ptr(), all_values.len());
            let _ = sums_tensor; // unused
        }

        let (output_indices_ptr, output_value_tensor) = value_tensor.topk_softmax(
            indices_ptr,
            output_sequences.as_mut_ptr(),
            batch_temperature.as_mut_ptr(),
            1,
            num_topk,
            1.0f16,
            0.0f16,
            false,
            vec![eos_id],
            "model.layers.0.topk_softmax".to_string(),
        );

        let operator_queue = f16::take_operator_queue();
        for i in 0..thread_num {
            for op in operator_queue.iter() {
                if let Operator::TopKSoftmax(operator) = op {
                    operator.run(
                        batch_size,
                        1,
                        thread_num,
                        i,
                        &[],
                        &decode_list,
                        &mut batch_list,
                    );
                } else {
                    op.run(batch_size, 1, thread_num, i, &[], &[], &mut Vec::new());
                }
            }
        }

        let num_tokens = batch_size;
        let output_indices =
            unsafe { std::slice::from_raw_parts(output_indices_ptr, num_tokens * num_topk) };
        let output_values =
            unsafe { std::slice::from_raw_parts(output_value_tensor.data, num_tokens * num_topk) };

        // token 0
        let mut candidates0: Vec<_> = indices0.iter().zip(values0.iter()).collect();
        candidates0.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values0: Vec<f32> = candidates0.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val0 = top_values0[0];
        let exps0: Vec<f32> = top_values0.iter().map(|v| (v - max_val0).exp()).collect();
        let sum_exps0: f32 = exps0.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[i], *candidates0[i].0);
            let val = output_values[i] as f32;
            let expected = exps0[i] / sum_exps0;
            assert!(
                (val - expected).abs() < 1e-3,
                "Mismatch at token 0 index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        // token 1
        let mut candidates1: Vec<_> = indices1.iter().zip(values1.iter()).collect();
        candidates1.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then_with(|| a.0.cmp(b.0)));

        let top_values1: Vec<f32> = candidates1.iter().take(num_topk).map(|c| *c.1).collect();
        let max_val1 = top_values1[0];
        let exps1: Vec<f32> = top_values1.iter().map(|v| (v - max_val1).exp()).collect();
        let sum_exps1: f32 = exps1.iter().sum();

        for i in 0..num_topk {
            assert_eq!(output_indices[num_topk + i], *candidates1[i].0);
            let val = output_values[num_topk + i] as f32;
            let expected = exps1[i] / sum_exps1;
            assert!(
                (val - expected).abs() < 1e-3,
                "Mismatch at token 1 index {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }

        assert_eq!(output_sequences[0], *candidates0[0].0);
        assert_eq!(output_sequences[1], *candidates1[0].0);
    }

    // ============================================================
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
        let q_weight =
            Tensor::<f16>::from_mem_pool(vec![q_dim, hidden_size], "q.weight".to_string());
        let k_weight =
            Tensor::<f16>::from_mem_pool(vec![kv_dim, hidden_size], "k.weight".to_string());
        let v_weight =
            Tensor::<f16>::from_mem_pool(vec![kv_dim, hidden_size], "v.weight".to_string());
        let q_norm_weight =
            Tensor::<f16>::from_mem_pool(vec![head_dim], "q_norm.weight".to_string());
        let k_norm_weight =
            Tensor::<f16>::from_mem_pool(vec![head_dim], "k_norm.weight".to_string());

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
                apply_qk_post_process_ref_f16(
                    &mut expected,
                    batch_size,
                    n_dim,
                    head_dim,
                    &rope_data,
                );
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

        let q_weight =
            Tensor::<f16>::from_mem_pool(vec![q_dim, hidden_size], "q.weight".to_string());
        let k_weight =
            Tensor::<f16>::from_mem_pool(vec![kv_dim, hidden_size], "k.weight".to_string());
        let v_weight =
            Tensor::<f16>::from_mem_pool(vec![kv_dim, hidden_size], "v.weight".to_string());
        let q_norm_weight =
            Tensor::<f16>::from_mem_pool(vec![head_dim], "q_norm.weight".to_string());
        let k_norm_weight =
            Tensor::<f16>::from_mem_pool(vec![head_dim], "k_norm.weight".to_string());

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

        let check_nt =
            |got: &[f16], w_nt: &[f16], n_dim: usize, name: &str, post_process_qk: bool| {
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
    // ExpertsMatMulSilu / ExpertsMatMulDown / ExpertsMergeAdd
    // weights now NT: [E, I, H] and [E, H, Hmid] respectively
    // ============================================================

    /// reference: out[e,b,i] = silu(sum_k a[b,k]*w_gate_nt[e,i,k]) * (sum_k a[b,k]*w_up_nt[e,i,k])
    fn ref_experts_silu_f32(
        a: &[f16],                  // [B,H]
        w_gate_nt: &[f16],          // [E,I,H] row-major
        w_up_nt: &[f16],            // [E,I,H]
        experts_indicator: &[bool], // [E]
        indice: &[bool],            // [E,B]
        out: &mut [f32],            // [E,B,I]
        b: usize,
        h: usize,
        i: usize,
        e: usize,
    ) {
        for v in out.iter_mut() {
            *v = 0.0;
        }

        for ex in 0..e {
            if !experts_indicator[ex] {
                continue;
            }
            for bb in 0..b {
                if !indice[ex * b + bb] {
                    continue;
                }
                for ii in 0..i {
                    let mut g = 0.0f32;
                    let mut u = 0.0f32;
                    for kk in 0..h {
                        let a_v = a[bb * h + kk] as f32;
                        let wg = w_gate_nt[ex * (i * h) + ii * h + kk] as f32;
                        let wu = w_up_nt[ex * (i * h) + ii * h + kk] as f32;
                        g += a_v * wg;
                        u += a_v * wu;
                    }
                    out[ex * (b * i) + bb * i + ii] = silu_f32(g) * u;
                }
            }
        }
    }

    #[inline]
    fn slot_of(topk: &[usize], b: usize, ktop: usize, e: usize) -> usize {
        let row = &topk[b * ktop..b * ktop + ktop];
        row.iter().position(|&x| x == e).unwrap_or(0)
    }

    fn ref_down_f32(
        nonlin: &[f16],   // [E,B,Hmid]
        wdown_nt: &[f16], // [E,H,Hmid] row-major (NT)
        experts_indicator: &[bool],
        indice: &[bool],     // [E,B]
        weight: &[f16],      // [E,B]
        topk: &[usize],      // [B,Ktop]
        out_ref: &mut [f32], // [B,Ktop,H]
        e: usize,
        b: usize,
        hmid: usize,
        h: usize,
        ktop: usize,
    ) {
        for v in out_ref.iter_mut() {
            *v = 0.0;
        }

        for ex in 0..e {
            if !experts_indicator[ex] {
                continue;
            }
            for bb in 0..b {
                if !indice[ex * b + bb] {
                    continue;
                }
                let s = slot_of(topk, bb, ktop, ex);
                let w = f32_from_f16(weight[ex * b + bb]);

                for j in 0..h {
                    let mut acc = 0.0f32;
                    for kk in 0..hmid {
                        let a = f32_from_f16(nonlin[(ex * b + bb) * hmid + kk]);
                        // NT: [j * hmid + kk]
                        let bj = f32_from_f16(wdown_nt[ex * (h * hmid) + j * hmid + kk]);
                        acc += a * bj;
                    }
                    out_ref[(bb * ktop + s) * h + j] += w * acc;
                }
            }
        }
    }

    #[test]
    fn test_experts_matmul_silu_f16_tensor_api() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }
        init_f16_tensor_test_runtime();

        let batch_size = 12;
        let hidden = 64; // H
        let inter = 64; // I
        let num_experts = 2;

        let input = Tensor::<f16>::from_mem_pool(
            vec![batch_size, hidden],
            "model.layers.0.input".to_string(),
        );

        // shape is [E, I, H], raw mem_mgr also [E, I, H] row-major (NT)
        let gate_w = Tensor::<f16>::from_mem_pool(
            vec![num_experts, inter, hidden],
            "gate.weight".to_string(),
        );
        let up_w =
            Tensor::<f16>::from_mem_pool(vec![num_experts, inter, hidden], "up.weight".to_string());

        let b = batch_size;

        let mut experts_box = AlignedBox::allocate_init(num_experts, false);
        let experts_indicator = experts_box.as_mut_ptr();
        std::mem::forget(experts_box);

        let mut indice_box = AlignedBox::allocate_init(num_experts * b, false);
        let indice_ptr = indice_box.as_mut_ptr();
        std::mem::forget(indice_box);

        unsafe {
            *experts_indicator.add(0) = true;
            *experts_indicator.add(1) = false;

            for bb in 0..b {
                *indice_ptr.add(0 * b + bb) = true;
                *indice_ptr.add(b + bb) = false;
            }
        }
        let mut routing_scores = vec![0.0f16; num_experts * b];
        let mut topk_indices = vec![0usize; b];
        for bb in 0..b {
            routing_scores[bb] = 1.0f16;
            topk_indices[bb] = 0;
        }
        let routing = unsafe {
            routing_from_dense(
                num_experts,
                b,
                1,
                indice_ptr,
                routing_scores.as_ptr(),
                topk_indices.as_ptr(),
            )
        };

        // input init
        let mut a = vec![0.0f16; b * hidden];
        for bb in 0..b {
            for kk in 0..hidden {
                a[bb * hidden + kk] = (((bb * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
            }
        }
        unsafe {
            input.data.copy_from_nonoverlapping(a.as_ptr(), a.len());
        }

        // weights init: [E, I, H] row-major
        let per_elems = inter * hidden;
        let mut wg_nt = vec![0.0f16; num_experts * per_elems];
        let mut wu_nt = vec![0.0f16; num_experts * per_elems];

        for e in 0..num_experts {
            for ii in 0..inter {
                for kk in 0..hidden {
                    let base_g = ((kk * 5 + ii * 11 + e * 13) % 23) as f32 * 0.01;
                    let base_u = ((kk * 9 + ii * 7 + e * 17) % 29) as f32 * 0.01;
                    wg_nt[e * per_elems + ii * hidden + kk] = base_g as f16;
                    wu_nt[e * per_elems + ii * hidden + kk] = base_u as f16;
                }
            }
        }

        unsafe {
            gate_w
                .data
                .copy_from_nonoverlapping(wg_nt.as_ptr(), wg_nt.len());
            up_w.data
                .copy_from_nonoverlapping(wu_nt.as_ptr(), wu_nt.len());
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let out = input.experts_matmul_silu_mul_matmul(
            &gate_w,
            &up_w,
            routing,
            params,
            false,
            "model.layers.0.experts_silu".to_string(),
        );

        assert_eq!(out.shape, vec![num_experts, batch_size, inter]);
        let op = take_single_f16_operator(|op| matches!(op, Operator::ExpertsMatMulSilu(_)));

        let thread_num = avail_threads();
        run_operator_all_threads(&op, b, 0, thread_num);

        let out_len = num_experts * b * inter;
        let out_got = unsafe { std::slice::from_raw_parts(out.data, out_len) };

        // reference for expert0
        for bb in 0..b {
            for ii in 0..inter {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for kk in 0..hidden {
                    let a_v = a[bb * hidden + kk] as f32;
                    let wg_v = wg_nt[0 * per_elems + ii * hidden + kk] as f32;
                    let wu_v = wu_nt[0 * per_elems + ii * hidden + kk] as f32;
                    g += a_v * wg_v;
                    u += a_v * wu_v;
                }
                let exp = silu_f32(g) * u;

                let got = out_got[0 * (b * inter) + bb * inter + ii] as f32;
                assert!(
                    (got - exp).abs() < 0.5,
                    "Mismatch expert0 bb {} ii {}: got {}, expected {}",
                    bb,
                    ii,
                    got,
                    exp
                );
            }
        }

        // expert1 inactive -> 0
        for bb in 0..b {
            for ii in 0..inter {
                let got = out_got[(b * inter) + bb * inter + ii] as f32;
                assert!(
                    got.abs() < 1e-3,
                    "Inactive expert1 should be ~0, but got {} at bb {} ii {}",
                    got,
                    bb,
                    ii
                );
            }
        }
    }

    #[test]
    fn test_experts_matmul_down_f16_tensor_api() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }
        init_f16_tensor_test_runtime();

        let batch_size = 12;
        let num_experts = 2;

        let inter = 64; // K (KC=64)
        let hidden = 32; // N (NR=32)
        let num_experts_per_tok = 1;

        let b = batch_size;

        // input to down: [E, seq, batch, inter]
        let x = Tensor::<f16>::from_mem_pool(
            vec![num_experts, batch_size, inter],
            "model.layers.0.experts.silu_out".to_string(),
        );

        // down weights: shape [E, hidden, inter]
        // ✅ NEW contract: B is already NT (N×K) row-major in mem_mgr per expert:
        // w_nt[j * inter + kk]
        let down_w = Tensor::<f16>::from_mem_pool(
            vec![num_experts, hidden, inter],
            "model.layers.0.down.weight".to_string(),
        );

        let mut experts_box = AlignedBox::allocate_init(num_experts, false);
        let experts_indicator = experts_box.as_mut_ptr();
        std::mem::forget(experts_box);

        let mut indice_box = AlignedBox::allocate_init(num_experts * b, false);
        let indice_ptr = indice_box.as_mut_ptr();
        std::mem::forget(indice_box);

        let mut weight_box = AlignedBox::allocate_init(num_experts * b, 0.0f16);
        let weight_ptr = weight_box.as_mut_ptr();
        std::mem::forget(weight_box);

        let mut topk_indices_box = AlignedBox::allocate_init(b * num_experts_per_tok, 0usize);
        let topk_indices_ptr = topk_indices_box.as_mut_ptr();
        std::mem::forget(topk_indices_box);

        unsafe {
            *experts_indicator.add(0) = true;
            *experts_indicator.add(1) = false;

            for t in 0..b {
                *indice_ptr.add(0 * b + t) = true;
                *indice_ptr.add(b + t) = false;

                *weight_ptr.add(0 * b + t) = 1.0f16;
                *weight_ptr.add(b + t) = 0.0f16;

                *topk_indices_ptr.add(t) = 0usize;
            }
        }

        // init X: expert0 pattern, expert1 zeros
        let mut x_e0 = vec![0.0f16; b * inter];
        for t in 0..b {
            for kk in 0..inter {
                x_e0[t * inter + kk] = (((t * 7 + kk * 3) % 19) as f32 * 0.01) as f16;
            }
        }
        unsafe {
            x.data
                .add(0 * (b * inter))
                .copy_from_nonoverlapping(x_e0.as_ptr(), x_e0.len());
            for i in 0..(b * inter) {
                *x.data.add((b * inter) + i) = 0.0f16;
            }
        }

        // init W_down:
        // ✅ NEW: per expert is NT (N×K) = hidden × inter row-major:
        // w_e0[j * inter + kk]
        let per_w = inter * hidden;
        let mut w_e0 = vec![0.0f16; per_w];
        let mut w_e1 = vec![0.0f16; per_w];

        for j in 0..hidden {
            for kk in 0..inter {
                // 跟以前一样的 deterministic pattern，只是存储索引变了
                w_e0[j * inter + kk] = (((kk * 5 + j * 11) % 23) as f32 * 0.01) as f16;
                w_e1[j * inter + kk] = (((kk * 3 + j * 7) % 29) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            down_w
                .data
                .add(0 * per_w)
                .copy_from_nonoverlapping(w_e0.as_ptr(), per_w);
            down_w
                .data
                .add(per_w)
                .copy_from_nonoverlapping(w_e1.as_ptr(), per_w);
        }

        let params = MatMulParams {
            a_row_step_macro: 6,
            b_row_step_macro: 64,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let out = x.experts_matmul_mul(
            &down_w,
            unsafe {
                routing_from_dense(
                    num_experts,
                    b,
                    num_experts_per_tok,
                    indice_ptr,
                    weight_ptr,
                    topk_indices_ptr,
                )
            },
            num_experts_per_tok,
            params,
            false,
            "model.layers.0.experts_down".to_string(),
        );

        assert_eq!(out.shape, vec![batch_size, num_experts_per_tok, hidden]);
        let op = take_single_f16_operator(|op| matches!(op, Operator::ExpertsMatMulDown(_)));

        // ✅ IMPORTANT: down does out += acc * factor, so zero out first
        let out_len = b * num_experts_per_tok * hidden;
        unsafe {
            for i in 0..out_len {
                *out.data.add(i) = 0.0f16;
            }
        }

        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        run_operator_all_threads(&op, b, 0, thread_num);

        // verify reference:
        // out[t, 0, j] = sum_k x_e0[t,k] * w_e0_nt[j,k]
        let out_got = unsafe { std::slice::from_raw_parts(out.data, out_len) };
        for t in 0..b {
            for j in 0..hidden {
                let mut sum = 0.0f32;
                for kk in 0..inter {
                    let x_v = x_e0[t * inter + kk] as f32;
                    let w_v = w_e0[j * inter + kk] as f32; // ✅ NT indexing
                    sum += x_v * w_v;
                }
                let got = out_got[t * hidden + j] as f32;
                assert!(
                    (got - sum).abs() < 0.5,
                    "Down mismatch token {} col {}: got {}, expected {}",
                    t,
                    j,
                    got,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_experts_merge_add_f16_tensor_api_k2_slot1_zero() {
        if !std::arch::is_x86_feature_detected!("avx512fp16") {
            println!("AVX512FP16 not supported, skipping test.");
            return;
        }
        init_f16_tensor_test_runtime();

        let batch_size = 12;
        let num_tokens = batch_size;

        let num_experts = 2; // 仅用于 reset gating（我们这里 reset_gating=false）
        let k = 2usize; // num_experts_per_token == K
        let hidden = 64usize;

        // input ptr layout: [num_tokens, K, H]
        let input = Tensor::<f16>::from_mem_pool(
            vec![batch_size, k, hidden],
            "model.layers.0.moe.down_out".to_string(),
        );

        let residual = Tensor::<f16>::from_mem_pool(
            vec![batch_size, hidden],
            "model.layers.0.residual".to_string(),
        );

        // routing buffers（reset_gating=false 不会用来选择，只会在 reset_gating=true 时清零）
        let mut experts_box = AlignedBox::allocate_init(num_experts, false);
        let experts_indicator = experts_box.as_mut_ptr();
        std::mem::forget(experts_box);

        let mut indice_box = AlignedBox::allocate_init(num_experts * num_tokens, false);
        let indice_ptr = indice_box.as_mut_ptr();
        std::mem::forget(indice_box);

        unsafe {
            *experts_indicator.add(0) = true;
            *experts_indicator.add(1) = true;
            for t in 0..num_tokens {
                *indice_ptr.add(0 * num_tokens + t) = true;
                *indice_ptr.add(num_tokens + t) = true;
            }
        }
        let mut routing_scores = vec![0.0f16; num_experts * num_tokens];
        let mut topk_indices = vec![0usize; num_tokens * k];
        for t in 0..num_tokens {
            routing_scores[t] = 1.0f16;
            routing_scores[num_tokens + t] = 1.0f16;
            topk_indices[t * k] = 0;
            topk_indices[t * k + 1] = 1;
        }
        let routing = unsafe {
            routing_from_dense(
                num_experts,
                num_tokens,
                k,
                indice_ptr,
                routing_scores.as_ptr(),
                topk_indices.as_ptr(),
            )
        };

        // init residual
        let mut r = vec![0.0f16; num_tokens * hidden];
        for t in 0..num_tokens {
            for h in 0..hidden {
                r[t * hidden + h] = (((t * 2 + h * 5) % 17) as f32 * 0.01) as f16;
            }
        }
        unsafe {
            residual.data.copy_from_nonoverlapping(r.as_ptr(), r.len());
        }

        // init input: slot0 = pattern, slot1 = 0
        let mut slot0 = vec![0.0f16; num_tokens * hidden];
        for t in 0..num_tokens {
            for h in 0..hidden {
                slot0[t * hidden + h] = (((t * 7 + h * 3) % 19) as f32 * 0.01) as f16;
            }
        }

        unsafe {
            // 全清零
            let total = num_tokens * k * hidden;
            for i in 0..total {
                *input.data.add(i) = 0.0f16;
            }
            // 写 slot0
            for t in 0..num_tokens {
                let base = t * (k * hidden);
                input
                    .data
                    .add(base + 0 * hidden)
                    .copy_from_nonoverlapping(slot0.as_ptr().add(t * hidden), hidden);
            }
        }

        // build via Tensor API
        let out = input.experts_merge_add(
            &residual,
            routing,
            false,
            "model.layers.0.experts_merge_add".to_string(),
        );

        assert_eq!(out.shape, vec![batch_size, hidden]);
        let op = take_single_f16_operator(|op| matches!(op, Operator::ExpertsMergeAdd(_)));
        match &op {
            Operator::ExpertsMergeAdd(operator) => {
                assert_eq!(operator.num_experts, num_experts);
                assert_eq!(operator.batch_size, batch_size);
            }
            _ => unreachable!(),
        }

        // run
        let thread_num = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        run_operator_all_threads(&op, num_tokens, 0, thread_num);

        // verify: out = residual + slot0 + slot1(0)
        let out_len = num_tokens * hidden;
        let out_got = unsafe { std::slice::from_raw_parts(out.data, out_len) };

        for t in 0..num_tokens {
            for h in 0..hidden {
                let exp = (r[t * hidden + h] as f32) + (slot0[t * hidden + h] as f32);
                let got = out_got[t * hidden + h] as f32;
                assert!(
                    (got - exp).abs() < 1e-3,
                    "MergeAdd mismatch token {} h {}: got {}, expected {}",
                    t,
                    h,
                    got,
                    exp
                );
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::GlobalOperatorQueue;
    use approx::assert_abs_diff_eq;
    use std::collections::HashMap;
    use std::f16;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

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

        let weight_tensor =
            Tensor::<f16>::from_mem_pool(vec![N, K], "perf.matmul.weight".to_string());

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
}
