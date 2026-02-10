use std::cell::RefCell;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

use serde::de;

use crate::kernel::generic::sigmoid::Sigmoid;
use crate::kernel::generic::sqrt::Sqrt;
use crate::kernel::generic::{exp::Exp, neg_infinity::NegInfinity};

use super::super::init::matmul_params::MatMulParams;
use super::super::memory::cache::Cache;
use super::super::ptensor::tensor::Tensor;
use crate::compiler::operator::Operator;

// use super::mlp::MLP;
// use super::super::ptensor::linear::Linear;

#[derive(Clone)]
pub struct SparseMoeBlock<T>
where
    T: Copy + PartialOrd,
{
    hidden_size: usize,
    num_experts: usize,
    num_topk: usize,
    norm_topk_prob: bool,
    gate_weight: Tensor<T>,
    experts_gate_weight: Tensor<T>,
    experts_up_weight: Tensor<T>,
    experts_down_weight: Tensor<T>,
    scope_name: String,
    cache: Rc<RefCell<Cache<T>>>,
    operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
}

impl<T> SparseMoeBlock<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid<T>
        + Sqrt
        + AddAssign,
{
    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        num_topk: usize,
        norm_topk_prob: bool,
        parent_scope_name: &str,
        cache: Rc<RefCell<Cache<T>>>,
        operator_queue: Rc<RefCell<Vec<Operator<T>>>>,
    ) -> Self {
        let scope_name = format!("{}.mlp", parent_scope_name);
        Self {
            // sequence_chunk_size,
            hidden_size,
            num_experts,
            num_topk,
            norm_topk_prob,
            gate_weight: Tensor::zeros(
                vec![num_experts, hidden_size],
                format!("{}.gate.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            experts_gate_weight: Tensor::zeros(
                vec![num_experts, moe_intermediate_size, hidden_size],
                format!("{}.experts.gate_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            experts_up_weight: Tensor::zeros(
                vec![num_experts, moe_intermediate_size, hidden_size],
                format!("{}.experts.up_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),

            experts_down_weight: Tensor::zeros(
                vec![num_experts, hidden_size, moe_intermediate_size],
                format!("{}.experts.down_proj.weight", scope_name),
                cache.clone(),
                operator_queue.clone(),
            ),
            scope_name: scope_name,
            cache: cache,
            operator_queue: operator_queue,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        decode_only_flag: bool,
        tensor_name: String,
    ) -> Tensor<T> {
        println!("Entering SparseMoeBlock forward: {}", tensor_name);
        println!("gate weight shape: {:?}", self.gate_weight.shape);
        // gate_output [sequence_chunk_size, batch_size, num_experts]
        let gate_output = hidden_states.matmul(
            &self.gate_weight,
            MatMulParams {
                a_row_step_macro: 3,
                b_row_step_macro: 128,
                column_step_macro: 16,
                a_row_step_micro: 3,
                b_row_step_micro: 32,
            },
            hidden_states.shape[0],
            decode_only_flag,
            format!("{}.gate", self.scope_name),
        );

        println!(
            "After gate matmul in SparseMoeBlock forward: {}",
            tensor_name
        );
        let (experts_indicator, indice_ptr, weight_ptr, topk_indices_ptr) = gate_output
            .experts_softmax_norm(
                self.num_experts,
                self.num_topk,
                decode_only_flag,
                format!("{}.router_probs", self.scope_name),
            );

        println!(
            "After experts_softmax_norm in SparseMoeBlock forward: {}",
            tensor_name
        );
        // nonlinear_product [num_experts, sequence_chunk_size, batch_size, intermediate_size]
        let nonlinear_product = hidden_states.experts_matmul_silu_mul_matmul(
            &self.experts_gate_weight,
            &self.experts_up_weight,
            experts_indicator,
            indice_ptr,
            MatMulParams {
                a_row_step_macro: 3,
                b_row_step_macro: 128,
                column_step_macro: 16,
                a_row_step_micro: 3,
                b_row_step_micro: 32,
            },
            decode_only_flag,
            format!("{}.gate_up", self.scope_name),
        );

        println!(
            "After experts_matmul_silu_mul_matmul in SparseMoeBlock forward: {}",
            tensor_name
        );
        // down_product [sequence_chunk_size, batch_size, num_experts_per_token, hidden_size]
        let down_product = nonlinear_product.experts_matmul_mul(
            &self.experts_down_weight,
            experts_indicator,
            indice_ptr,
            weight_ptr,
            topk_indices_ptr,
            self.num_topk,
            MatMulParams {
                a_row_step_macro: 3,
                b_row_step_macro: 128,
                column_step_macro: 16,
                a_row_step_micro: 3,
                b_row_step_micro: 32,
            },
            decode_only_flag,
            format!("{}.down", self.scope_name),
        );

        println!(
            "After experts_matmul_mul in SparseMoeBlock forward: {}",
            tensor_name
        );

        let merge_tensor = down_product.experts_merge_add(
            residual,
            experts_indicator,
            indice_ptr,
            self.num_experts,
            decode_only_flag,
            format!("{}.merge", self.scope_name),
        );
        merge_tensor
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compiler::operator::Operator;
    use std::cell::RefCell;
    use std::mem;
    use std::rc::Rc;

    // ---------- helpers (match your project style: transmute for f16 bits) ----------

    #[inline]
    fn f16_bits(x: f16) -> u16 {
        unsafe { mem::transmute::<f16, u16>(x) }
    }

    #[inline]
    fn f16_from_bits(bits: u16) -> f16 {
        unsafe { mem::transmute::<u16, f16>(bits) }
    }

    #[inline]
    fn f16_one() -> f16 {
        (1.0f32) as f16
        // 等价：f16_from_bits(0x3C00)
    }

    #[inline]
    fn f16_seven() -> f16 {
        (7.0f32) as f16
        // 等价：f16_from_bits(0x4700)
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
                // 本次不考虑 position，按你现有约定固定 (0,1)
                op.run(batch_size, 0, thread_num, tid, &[], &[], &mut Vec::new());
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
    ) -> (SparseMoeBlock<f16>, Tensor<f16>, Tensor<f16>) {
        let cache = Rc::new(RefCell::new(Cache::<f16>::new(
            std::collections::HashMap::new(),
        )));
        let operator_queue = Rc::new(RefCell::new(Vec::<Operator<f16>>::new()));

        let moe = SparseMoeBlock::<f16>::new(
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            true,
            "model.layers.0",
            cache.clone(),
            operator_queue.clone(),
        );

        let shape = vec![sequence_chunk_size, batch_size, hidden_size];

        let input = Tensor::from_cache(
            shape.clone(),
            "model.layers.0.input_tensor".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        let residual = Tensor::from_cache(
            shape.clone(),
            "model.layers.0.residual_tensor".to_string(),
            cache.clone(),
            operator_queue.clone(),
        );

        (moe, input, residual)
    }

    // ---------- tests ----------

    #[test]
    fn test_sparse_moe_queue_structure() {
        let (moe, input, residual) = build_case(1, 24, 256, 1024, 128, 8);

        let out = moe.forward(
            &input,
            &residual,
            false,
            "model.layers.0.output".to_string(),
        );
        let q = out.operator_queue.borrow();

        assert!(q.len() >= 5, "Expected >=5 operators, got {}", q.len());

        // 如果你的 Operator enum 名字不同，只需要改下面 match 的分支名
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
    fn test_sparse_moe_zero_weights_output_equals_residual_bits() {
        let (moe, input, residual) = build_case(1, 24, 256, 1024, 128, 8);

        // 权重全 0：down 分支应为 0 -> output 应严格等于 residual（bit-level）
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
        let (moe, input, residual) = build_case(1, 24, 256, 1024, 128, 8);

        fill_tensor_f16(&input, f16_one());
        fill_tensor_f16(&residual, f16_one());

        let out = moe.forward(
            &input,
            &residual,
            false,
            "model.layers.0.output".to_string(),
        );

        // 单线程
        run_queue(&out, 24, 1);
        let out1_bits = tensor_to_bits_vec(&out);

        // 防止假阳性：先把 out buffer 写成明显值，再跑多线程
        fill_tensor_f16(&out, f16_seven());

        // 多线程
        run_queue(&out, 24, num_cpus::get());
        let outn_bits = tensor_to_bits_vec(&out);

        assert_eq!(
            out1_bits, outn_bits,
            "single-thread != multi-thread (bit-level)"
        );
    }
}
