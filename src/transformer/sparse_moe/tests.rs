use std::cell::RefCell;
use std::rc::Rc;

use crate::mem_mgr::cache::Cache;
use crate::runtime::operator::Operator;
use crate::runtime::tensor::{Tensor, TensorCtx};
use crate::transformer::config::RouterScoringKind;
use crate::transformer::names::SparseMoeTensorNames;

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

fn run_queue(output: &Tensor<f16>, batch_size: usize, thread_num: usize) {
    for op in output.operator_queue.borrow().iter() {
        for tid in 0..thread_num {
            op.run(
                batch_size,
                0,
                thread_num,
                tid,
                &[],
                &[],
                &[],
                &mut Vec::new(),
            );
        }
    }
}

fn build_case(
    sequence_chunk_size: usize,
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    router_scoring: RouterScoringKind,
    use_routing_bias: bool,
) -> (SparseMoe<f16>, Tensor<f16>, Tensor<f16>) {
    let cache = Rc::new(RefCell::new(Cache::<f16>::new(
        std::collections::HashMap::new(),
    )));
    let operator_queue = Rc::new(RefCell::new(Vec::<Operator<f16>>::new()));
    let ctx = Rc::new(TensorCtx::new(cache, operator_queue));

    let moe = SparseMoe::<f16>::new(
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        true,
        router_scoring,
        use_routing_bias,
        SparseMoeTensorNames {
            scope: String::from("model.layers.0.mlp"),
            router_gate: String::from("model.layers.0.mlp.gate.weight"),
            router_bias: None,
            experts_gate_proj: String::from("model.layers.0.mlp.experts.gate_proj.weight"),
            experts_up_proj: String::from("model.layers.0.mlp.experts.up_proj.weight"),
            experts_down_proj: String::from("model.layers.0.mlp.experts.down_proj.weight"),
        },
        ctx.clone(),
    );

    let shape = vec![sequence_chunk_size, batch_size, hidden_size];
    let input = ctx.tensor(shape.clone(), "model.layers.0.input_tensor".to_string());
    let residual = ctx.tensor(shape.clone(), "model.layers.0.residual_tensor".to_string());

    (moe, input, residual)
}

#[test]
fn test_sparse_moe_queue_structure() {
    let (moe, input, residual) =
        build_case(1, 24, 256, 1024, 128, 8, RouterScoringKind::Softmax, false);

    let out = moe.forward(
        &input,
        &residual,
        false,
        "model.layers.0.output".to_string(),
    );
    let q = out.operator_queue.borrow();

    assert!(q.len() >= 5, "Expected >=5 operators, got {}", q.len());

    match &q[0] {
        Operator::MatMul(_) => {}
        _ => panic!("op[0] should be MatMul"),
    }
    match &q[1] {
        Operator::ExpertsSoftmaxNorm(_) => {}
        _ => panic!("op[1] should be ExpertsSoftmaxNorm"),
    }
    match &q[2] {
        Operator::ExpertsMatMulSilu(_) => {}
        _ => panic!("op[2] should be ExpertsMatMulSilu"),
    }
    match &q[3] {
        Operator::ExpertsMatMulDown(_) => {}
        _ => panic!("op[3] should be ExpertsMatMulDown"),
    }
    match &q[4] {
        Operator::ExpertsMergeAdd(_) => {}
        _ => panic!("op[4] should be ExpertsMergeAdd"),
    }
}

#[test]
fn test_sparse_moe_sigmoid_queue_structure() {
    let (moe, input, residual) =
        build_case(1, 24, 256, 1024, 128, 8, RouterScoringKind::Sigmoid, true);

    let out = moe.forward(
        &input,
        &residual,
        false,
        "model.layers.0.output".to_string(),
    );
    let q = out.operator_queue.borrow();

    assert!(q.len() >= 5, "Expected >=5 operators, got {}", q.len());

    match &q[0] {
        Operator::MatMulSigmoid(_) => {}
        _ => panic!("op[0] should be MatMulSigmoid"),
    }
    match &q[1] {
        Operator::ExpertsTopkNorm(_) => {}
        _ => panic!("op[1] should be ExpertsTopkNorm"),
    }
    match &q[2] {
        Operator::ExpertsMatMulSilu(_) => {}
        _ => panic!("op[2] should be ExpertsMatMulSilu"),
    }
    match &q[3] {
        Operator::ExpertsMatMulDown(_) => {}
        _ => panic!("op[3] should be ExpertsMatMulDown"),
    }
    match &q[4] {
        Operator::ExpertsMergeAdd(_) => {}
        _ => panic!("op[4] should be ExpertsMergeAdd"),
    }
}

#[test]
fn test_sparse_moe_zero_weights_output_equals_residual_bits() {
    let (moe, input, residual) =
        build_case(1, 24, 256, 1024, 128, 8, RouterScoringKind::Softmax, false);

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
fn test_sparse_moe_single_thread_equals_multi_thread_bits() {
    let (moe, input, residual) =
        build_case(1, 24, 256, 1024, 128, 8, RouterScoringKind::Softmax, false);

    fill_tensor_f16(&input, f16_one());
    fill_tensor_f16(&residual, f16_one());

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
