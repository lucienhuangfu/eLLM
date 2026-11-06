use crate::config::TextConfig;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::operators::operator::Operator;
use crate::runtime::SequenceSlice;
use crate::tensor::{GlobalOperatorQueue, Tensor};
use std::collections::HashMap;

const EMPTY_SLICES: &[SequenceSlice] = &[];

use super::SparseMoe;

#[inline]
fn f16_bits(x: f16) -> u16 {
    x.to_bits()
}

#[inline]
fn f16_one() -> f16 {
    (1.0f32) as f16
}

#[inline]
fn f16_seven() -> f16 {
    (7.0f32) as f16
}

fn fill_tensor_f16(t: &Tensor<f16>, v: f16) {
    let n: usize = t.shape.iter().product();
    for i in 0..n {
        unsafe { t.data.add(i).write(v) }
    }
}

fn tensor_to_bits_vec(t: &Tensor<f16>) -> Vec<u16> {
    let n: usize = t.shape.iter().product();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = unsafe { *t.data.add(i) };
        out.push(f16_bits(x));
    }
    out
}

fn run_queue(_output: &Tensor<f16>, batch_size: usize, thread_num: usize) {
    f16::with_operator_queue(|queue| {
        for op in queue.iter() {
            for tid in 0..thread_num {
                op.run(
                    batch_size,
                    0,
                    0,
                    batch_size,
                    thread_num,
                    tid,
                    EMPTY_SLICES,
                    &mut Vec::new(),
                );
            }
        }
    });
}

/// Build a `TextConfig` for the sparse-MoE variant under test. Only the MoE-relevant
/// fields matter; everything else takes `Default::default()`.
/// 为被测 sparse-MoE 变体构造 `TextConfig`，仅 MoE 相关字段有意义。
fn moe_config(
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    scoring_func: &str,
    use_routing_bias: bool,
    shared_intermediate_size: usize,
) -> TextConfig {
    TextConfig {
        model_type: "qwen3_moe".to_string(),
        hidden_size,
        num_experts,
        num_experts_per_tok: top_k,
        moe_intermediate_size: intermediate_size,
        intermediate_size,
        shared_intermediate_size,
        scoring_func: scoring_func.to_string(),
        use_routing_bias,
        decoder_sparse_step: 1,
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn build_case(
    sequence_length: usize,
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    scoring_func: &str,
    use_routing_bias: bool,
    shared_intermediate_size: usize,
) -> (SparseMoe<f16>, Tensor<f16>, Tensor<f16>) {
    f16::init_global(HashMap::new());
    f16::init_operator_queue();

    let config = moe_config(
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        scoring_func,
        use_routing_bias,
        shared_intermediate_size,
    );
    let moe = SparseMoe::<f16>::new(&config, "model.layers.0.mlp");

    let shape = vec![sequence_length * batch_size, hidden_size];
    let input = Tensor::from_mem_pool(shape.clone(), "model.layers.0.input_tensor".to_string());
    let residual =
        Tensor::from_mem_pool(shape.clone(), "model.layers.0.residual_tensor".to_string());

    (moe, input, residual)
}

#[test]
fn test_sparse_moe_queue_structure() {
    // Routed-only (shared_intermediate_size == 0): router (2) + fused MLP + routed merge.
    // routed-only（shared_intermediate_size == 0）：router（2 个）+ 融合 MLP + routed merge。
    let (moe, input, residual) = build_case(1, 24, 256, 1024, 128, 8, "softmax", false, 0);

    let _out = moe.forward(
        &input,
        &residual,
        false,
        "model.layers.0.output".to_string(),
    );
    f16::with_operator_queue(|q| {
        assert_eq!(q.len(), 4, "Expected 4 operators, got {}", q.len());

        match &q[0] {
            Operator::MatMul(_) => {}
            _ => panic!("op[0] should be MatMul"),
        }
        match &q[1] {
            Operator::ExpertSoftmaxNorm(_) => {}
            _ => panic!("op[1] should be ExpertSoftmaxNorm"),
        }
        match &q[2] {
            Operator::SharedExpertMLP(_) => {}
            _ => panic!("op[2] should be SharedExpertMLP"),
        }
        match &q[3] {
            Operator::SharedExpertMergeAdd(_) => {}
            _ => panic!("op[3] should be SharedExpertMergeAdd"),
        }
    });
}

#[test]
fn test_sparse_moe_sigmoid_queue_structure() {
    let (moe, input, residual) = build_case(1, 24, 256, 1024, 128, 8, "sigmoid", true, 0);

    let _out = moe.forward(
        &input,
        &residual,
        false,
        "model.layers.0.output".to_string(),
    );
    f16::with_operator_queue(|q| {
        assert_eq!(q.len(), 4, "Expected 4 operators, got {}", q.len());

        match &q[0] {
            Operator::MatMulSigmoid(_) => {}
            _ => panic!("op[0] should be MatMulSigmoid"),
        }
        match &q[1] {
            Operator::ExpertTopkNorm(_) => {}
            _ => panic!("op[1] should be ExpertTopkNorm"),
        }
        match &q[2] {
            Operator::SharedExpertMLP(_) => {}
            _ => panic!("op[2] should be SharedExpertMLP"),
        }
        match &q[3] {
            Operator::SharedExpertMergeAdd(_) => {}
            _ => panic!("op[3] should be SharedExpertMergeAdd"),
        }
    });
}

#[test]
fn test_sparse_moe_shared_queue_structure() {
    // Shared-expert variant: router (2) + fused MLP + shared merge-add.
    // 共享专家变体：router（2 个）+ 融合 MLP + shared merge-add。
    let (moe, input, residual) = build_case(1, 24, 256, 1024, 128, 8, "softmax", false, 512);

    let _out = moe.forward(
        &input,
        &residual,
        false,
        "model.layers.0.output".to_string(),
    );
    f16::with_operator_queue(|q| {
        assert_eq!(q.len(), 4, "Expected 4 operators, got {}", q.len());

        match &q[2] {
            Operator::SharedExpertMLP(_) => {}
            _ => panic!("op[2] should be SharedExpertMLP"),
        }
        match &q[3] {
            Operator::SharedExpertMergeAdd(_) => {}
            _ => panic!("op[3] should be SharedExpertMergeAdd"),
        }
    });
}

#[test]
fn test_sparse_moe_zero_weights_output_equals_residual_bits() {
    // Shared-expert variant with zero weights: routed slots = 0 and gated shared
    // (sigmoid(0)·shared=0.5·0=0), so output == residual at bit level.
    // 共享专家零权重：routed slot 全 0，门控 shared = sigmoid(0)·0 = 0，输出等于 residual。
    let (moe, input, residual) = build_case(1, 24, 256, 1024, 128, 8, "softmax", false, 512);

    fill_tensor_f16(&input, f16_one());
    fill_tensor_f16(&residual, f16_one());

    let out = moe.forward(
        &input,
        &residual,
        false,
        "model.layers.0.output".to_string(),
    );

    run_queue(&out, 24, num_cpus::get());

    let out_bits = tensor_to_bits_vec(&out);
    let res_bits = tensor_to_bits_vec(&residual);
    assert_eq!(out_bits, res_bits, "output != residual (bit-level)");
}

#[test]
fn test_sparse_moe_routed_only_null_shared_output_equals_residual() {
    // Routed-only via the SINGLE fused merge (shared weights = None → null pointers):
    // the operator must skip the gated shared term without dereferencing null.
    // 单一融合 merge 走 routed-only（shared=None → 空指针）：算子必须跳过带门控 shared 项，
    // 不解引用空指针。
    let (moe, input, residual) = build_case(1, 24, 256, 1024, 128, 8, "softmax", false, 0);

    fill_tensor_f16(&input, f16_one());
    fill_tensor_f16(&residual, f16_one());

    let out = moe.forward(
        &input,
        &residual,
        false,
        "model.layers.0.output".to_string(),
    );

    run_queue(&out, 24, num_cpus::get());

    let out_bits = tensor_to_bits_vec(&out);
    let res_bits = tensor_to_bits_vec(&residual);
    assert_eq!(
        out_bits, res_bits,
        "routed-only fused merge: output != residual (bit-level)"
    );
}

#[test]
fn test_sparse_moe_single_thread_equals_multi_thread_bits() {
    let (moe, input, residual) = build_case(1, 24, 256, 1024, 128, 8, "softmax", false, 512);

    fill_tensor_f16(&input, f16_one());
    fill_tensor_f16(&residual, f16_one());

    // Non-trivial shared/routed weights so the merge actually sums values.
    // 填充非零 shared/routed 权重，使 merge 真正累加数值。`tests` 是 `sparse_moe`
    // 的子模块，可直接访问 `SparseMoe` 的私有字段。
    fill_tensor_f16(&moe.experts_gate_weight, f16_one());
    fill_tensor_f16(&moe.experts_up_weight, f16_one());
    fill_tensor_f16(&moe.experts_down_weight, f16_one());
    fill_tensor_f16(
        &moe.shared_gate_weight
            .as_ref()
            .expect("shared weight present when shared_intermediate_size > 0"),
        f16_one(),
    );
    fill_tensor_f16(moe.shared_up_weight.as_ref().unwrap(), f16_one());
    fill_tensor_f16(moe.shared_down_weight.as_ref().unwrap(), f16_one());
    fill_tensor_f16(moe.shared_expert_gate_weight.as_ref().unwrap(), f16_one());

    let out = moe.forward(
        &input,
        &residual,
        false,
        "model.layers.0.output".to_string(),
    );

    run_queue(&out, 24, 1);
    let out1_bits = tensor_to_bits_vec(&out);

    fill_tensor_f16(&out, f16_seven());

    run_queue(&out, 24, num_cpus::get());
    let outn_bits = tensor_to_bits_vec(&out);

    assert_eq!(
        out1_bits, outn_bits,
        "single-thread != multi-thread (bit-level)"
    );
}
