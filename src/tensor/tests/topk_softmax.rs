use super::common::*;
use super::*;
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
    let sums_tensor = Tensor::<f32>::from_mem_pool(sums_shape, "model.layers.0.sums".to_string());

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
    let sums_tensor = Tensor::<f16>::from_mem_pool(sums_shape, "model.layers.0.sums".to_string());

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
