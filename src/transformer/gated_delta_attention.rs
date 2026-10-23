use std::ops::{AddAssign, Neg, Sub};

use crate::kernel::common::matmul_params::MatMulParams;
use crate::mem_mgr::mem_pool::GlobalMemPool;
use crate::num_traits::{Exp, FromNumber, NegInfinity, Sigmoid, Sqrt};
use crate::tensor::{GlobalOperatorQueue, Tensor};

use super::tensor_name::GatedDeltaAttentionTensorNames;

// Qwen3.5-MoE gated delta linear attention block
// (scripts/modeling_qwen3_5_moe.py::Qwen3_5MoeGatedDeltaNet), a two-stage
// pipeline: FusedGatedDeltaRule (hidden -> four input projections qkv / z /
// b / a with the beta/g epilogue -> depthwise causal conv1d + SiLU ->
// qk-norm -> chunked gated delta rule in a single interleaved loop ->
// gated RMSNorm -> attention output) -> MatMulAdd (out_proj + residual).
// The legacy three-operator path (MatMulProj -> CausalConv1dSilu ->
// RecurrentGatedDeltaRule) is still in the library, unused by this module.
// Qwen3.5-MoE 的 GatedDeltaAttention 线性注意力块
// （scripts/modeling_qwen3_5_moe.py::Qwen3_5MoeGatedDeltaNet），
// 两级流水线：FusedGatedDeltaRule（hidden -> 四路输入投影 qkv / z / b / a
// 含 beta/g epilogue -> depthwise 因果卷积 + SiLU -> qk 归一化 ->
// 单循环交错的 chunk 化 gated delta rule 递推 -> gated RMSNorm ->
// 注意力输出）-> MatMulAdd（out_proj + residual）。
// 旧三算子路径（MatMulProj -> CausalConv1dSilu ->
// RecurrentGatedDeltaRule）仍保留在库中，本模块不再调用。

// Delta rule chunking constant, aligned with the reference default.
// delta rule 分块常量，对齐参考实现默认值。
const FUSED_CHUNK_SIZE: usize = 64;

#[derive(Clone)]
pub struct GatedDeltaAttention<T>
where
    T: Copy + PartialOrd,
{
    sequence_length: usize,
    batch_size: usize,
    // Configuration dims kept on the struct; the fused path re-derives them
    // from the weight/state shapes inside the builder.
    // 结构体保留的配置维度；融合路径在 builder 内从权重/状态形状
    // 重新推导。
    #[allow(dead_code)]
    key_dim: usize,
    value_dim: usize,
    #[allow(dead_code)]
    head_k_dim: usize,
    qkv_weight: Tensor<T>,
    z_weight: Tensor<T>,
    b_weight: Tensor<T>,
    a_weight: Tensor<T>,
    dt_bias: Tensor<T>,
    a_log: Tensor<T>,
    conv_weight: Tensor<T>,
    // Per-head weight of the gated RMSNorm applied before out_proj; the norm
    // itself runs inside the FusedGatedDeltaRule epilogue, this handle
    // feeds the operator and keeps the HF key (…linear_attn.norm.weight)
    // loadable.
    // out_proj 前逐头 gated RMSNorm 的权重；归一化本体融合在
    // FusedGatedDeltaRule 的 epilogue 中，此句柄传入融合算子，并按
    // HF 键名加载权重。
    norm_weight: Tensor<T>,
    out_weight: Tensor<T>,
    // Rolling conv window in the fused per-task layout
    // [batch_size, num_v_heads, 2 * head_k_dim + head_v_dim, kernel_size - 1],
    // in-segment order q | k | v; only FusedGatedDeltaRule consumes it.
    // 融合路径逐任务布局的卷积滚动窗口
    // [batch_size, num_v_heads, 2 * head_k_dim + head_v_dim, kernel_size - 1]，
    // 段内顺序 q | k | v；仅 FusedGatedDeltaRule 消费。
    conv_state: Tensor<T>,
    // Recurrent cache [batch_size, num_v_heads, head_k_dim, head_v_dim].
    // 递推状态缓存 [batch_size, num_v_heads, head_k_dim, head_v_dim]。
    recurrent_state: Tensor<T>,
    scope_name: String,
}

impl<T> GatedDeltaAttention<T>
where
    T: Copy
        + PartialOrd
        + Default
        + Sub<Output = T>
        + Neg<Output = T>
        + Exp
        + NegInfinity
        + Sigmoid
        + Sqrt
        + FromNumber
        + AddAssign
        + GlobalMemPool
        + GlobalOperatorQueue,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
        sequence_length: usize,
        batch_size: usize,
        names: GatedDeltaAttentionTensorNames,
    ) -> Self {
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;

        Self {
            sequence_length,
            batch_size,
            key_dim,
            value_dim,
            head_k_dim,
            qkv_weight: Tensor::zeros(vec![conv_dim, hidden_size], names.in_proj_qkv),
            z_weight: Tensor::zeros(vec![value_dim, hidden_size], names.in_proj_z),
            b_weight: Tensor::zeros(vec![num_v_heads, hidden_size], names.in_proj_b),
            a_weight: Tensor::zeros(vec![num_v_heads, hidden_size], names.in_proj_a),
            dt_bias: Tensor::zeros(vec![num_v_heads], names.dt_bias),
            a_log: Tensor::zeros(vec![num_v_heads], names.a_log),
            conv_weight: Tensor::zeros(vec![conv_dim, conv_kernel_size], names.conv1d),
            norm_weight: Tensor::zeros(vec![head_v_dim], names.norm),
            out_weight: Tensor::zeros(vec![hidden_size, value_dim], names.out_proj),
            conv_state: Tensor::zeros(
                vec![
                    batch_size,
                    num_v_heads,
                    head_k_dim * 2 + head_v_dim,
                    conv_kernel_size - 1,
                ],
                format!("{}.conv_state", names.scope),
            ),
            recurrent_state: Tensor::zeros(
                vec![batch_size, num_v_heads, head_k_dim, head_v_dim],
                format!("{}.recurrent_state", names.scope),
            ),
            scope_name: names.scope,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<T>,
        residual: &Tensor<T>,
        decode_only_flag: bool,
        _tensor_name: String,
    ) -> Tensor<T> {
        // Fused linear attention: four-way projection (with the beta/g
        // epilogue), causal conv1d + SiLU, qk-norm, chunked gated delta
        // rule and the gated RMSNorm all run inside one operator over
        // (batch, v_head) tasks; both caches are updated in place and the
        // result is [sequence_length, batch_size, value_dim].
        // 融合线性注意力：四路投影（含 beta/g epilogue）、因果卷积 +
        // SiLU、qk 归一化、chunk 化 gated delta rule 与 gated RMSNorm
        // 全部在单算子内按 (batch, v_head) 任务完成；两份缓存原地
        // 更新，输出 [sequence_length, batch_size, value_dim]。
        let thread_num = num_cpus::get().max(1);
        let attn_output = hidden_states.fused_gated_delta_rule(
            &self.qkv_weight,
            &self.z_weight,
            &self.b_weight,
            &self.a_weight,
            &self.dt_bias,
            &self.a_log,
            &self.conv_weight,
            &self.norm_weight,
            &self.conv_state,
            &self.recurrent_state,
            FUSED_CHUNK_SIZE,
            thread_num,
            self.sequence_length,
            self.batch_size,
            self.scope_name.clone(),
        );

        // out_proj + residual.
        let output_sequence_length = attn_output.shape[0];
        let output_batch_size = attn_output.shape[1];
        let output_rows = output_sequence_length * output_batch_size;

        let view_attn_output = attn_output.view(vec![output_rows, self.value_dim]);
        if decode_only_flag {
            view_attn_output.lift_vector();
        }
        let residual_hidden_size = *residual
            .shape
            .last()
            .expect("residual tensor must have at least one dimension");
        let view_residual = residual.view(vec![output_rows, residual_hidden_size]);

        let output_2d = view_attn_output.matmul_add(
            &self.out_weight,
            &view_residual,
            MatMulParams {
                a_row_step_macro: 1,
                b_row_step_macro: 64,
                column_step_macro: 1,
                a_row_step_micro: 1,
                b_row_step_micro: 1,
            },
            decode_only_flag,
            format!("{}.output", self.scope_name),
        );
        output_2d.view(vec![
            output_sequence_length,
            output_batch_size,
            self.out_weight.shape[0],
        ])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::runtime::SequenceSlice;
    use std::collections::HashMap;

    const EMPTY_SLICES: &[SequenceSlice] = &[];

    #[test]
    fn test_gated_delta_attention() {
        const HIDDEN: usize = 8;
        const SEQUENCE_LENGTH: usize = 2;
        const BATCH_SIZE: usize = 1;
        // num_k_heads = 1, head_k_dim = 2 -> key_dim = 2;
        // num_v_heads = 2, head_v_dim = 2 -> value_dim = 4;
        // conv_dim = 2 * key_dim + value_dim = 8.
        const NUM_K_HEADS: usize = 1;
        const NUM_V_HEADS: usize = 2;
        const HEAD_K_DIM: usize = 2;
        const HEAD_V_DIM: usize = 2;
        const KERNEL_SIZE: usize = 4;

        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        let scope = String::from("model.layers.0.linear_attn");
        let module = GatedDeltaAttention::<f32>::new(
            HIDDEN,
            NUM_K_HEADS,
            NUM_V_HEADS,
            HEAD_K_DIM,
            HEAD_V_DIM,
            KERNEL_SIZE,
            SEQUENCE_LENGTH,
            BATCH_SIZE,
            GatedDeltaAttentionTensorNames {
                scope: scope.clone(),
                in_proj_qkv: format!("{}.in_proj_qkv.weight", scope),
                in_proj_z: format!("{}.in_proj_z.weight", scope),
                in_proj_b: format!("{}.in_proj_b.weight", scope),
                in_proj_a: format!("{}.in_proj_a.weight", scope),
                dt_bias: format!("{}.dt_bias", scope),
                a_log: format!("{}.a_log", scope),
                conv1d: format!("{}.conv1d.weight", scope),
                norm: format!("{}.norm.weight", scope),
                out_proj: format!("{}.out_proj.weight", scope),
            },
        );

        let hidden_states = Tensor::zeros(
            vec![SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN],
            String::from("model.layers.0.hidden_states"),
        );
        let residual = Tensor::zeros(
            vec![SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN],
            String::from("model.layers.0.residual"),
        );

        let output = module.forward(
            &hidden_states,
            &residual,
            false,
            String::from("test_output"),
        );
        debug_assert_eq!(output.shape, vec![SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN]);

        // Drain the operator queue: the fused pipeline must enqueue exactly
        // two operators, FusedGatedDeltaRule followed by MatMulAdd.
        // 排空算子队列：融合流水线应恰好入队两个算子，
        // FusedGatedDeltaRule 后接 MatMulAdd。
        let thread_num: usize = num_cpus::get();
        f32::with_operator_queue(|queue| {
            assert_eq!(queue.len(), 2);
            assert_eq!(queue[0].kind(), "FusedGatedDeltaRule");
            assert_eq!(queue[1].kind(), "MatMulAdd");
            for operator in queue.iter() {
                for i in 0..thread_num {
                    operator.run(
                        BATCH_SIZE,
                        0,
                        0,
                        0,
                        thread_num,
                        i,
                        EMPTY_SLICES,
                        &mut Vec::new(),
                    );
                }
            }
        });
    }
}
