use super::*;

// Enqueue-level tests for the GatedDeltaNet linear attention tensor API:
// each builder must allocate the documented outputs and push exactly one
// operator of the expected variant onto the global queue.
// GatedDeltaNet 线性注意力 Tensor API 的入队级测试：
// 每个 builder 必须按约定分配输出，并向全局队列压入恰好一个预期变体的算子。

const HIDDEN: usize = 8;
const SEQUENCE_LENGTH: usize = 2;
const BATCH_SIZE: usize = 1;

// qkv segment: key_dim * 2 + value_dim = 4 * 2 + 4 = 12.
const QKV_COLS: usize = 12;
const VALUE_DIM: usize = 4;
const HEAD_COLS: usize = 2;
const KEY_DIM: usize = 4;
const HEAD_K_DIM: usize = 2;

fn init_f32_test() {
    f32::init_global(HashMap::new());
    f32::init_operator_queue();
}

fn take_single_f32_operator() -> Operator<f32> {
    let queue = f32::take_operator_queue();
    assert_eq!(queue.len(), 1);
    queue.into_iter().next().unwrap()
}

fn proj_inputs() -> (
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
) {
    let hidden_states = Tensor::<f32>::from_mem_pool(
        vec![SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN],
        "model.layers.0.hidden_states".to_string(),
    );
    let qkv_weight = Tensor::<f32>::from_mem_pool(
        vec![QKV_COLS, HIDDEN],
        "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
    );
    let z_weight = Tensor::<f32>::from_mem_pool(
        vec![VALUE_DIM, HIDDEN],
        "model.layers.0.linear_attn.in_proj_z.weight".to_string(),
    );
    let b_weight = Tensor::<f32>::from_mem_pool(
        vec![HEAD_COLS, HIDDEN],
        "model.layers.0.linear_attn.in_proj_b.weight".to_string(),
    );
    let a_weight = Tensor::<f32>::from_mem_pool(
        vec![HEAD_COLS, HIDDEN],
        "model.layers.0.linear_attn.in_proj_a.weight".to_string(),
    );
    let dt_bias = Tensor::<f32>::from_mem_pool(
        vec![HEAD_COLS],
        "model.layers.0.linear_attn.dt_bias".to_string(),
    );
    let a_log = Tensor::<f32>::from_mem_pool(
        vec![HEAD_COLS],
        "model.layers.0.linear_attn.a_log".to_string(),
    );
    (
        hidden_states,
        qkv_weight,
        z_weight,
        b_weight,
        a_weight,
        dt_bias,
        a_log,
    )
}

#[test]
fn test_matmul_proj_enqueue() {
    init_f32_test();

    let (hidden_states, qkv_weight, z_weight, b_weight, a_weight, dt_bias, a_log) = proj_inputs();
    let input_rows = SEQUENCE_LENGTH * BATCH_SIZE;

    let (qkv_state, z_state, beta_state, g_state) = hidden_states.matmul_proj(
        &qkv_weight,
        &z_weight,
        &b_weight,
        &a_weight,
        &dt_bias,
        &a_log,
        SEQUENCE_LENGTH,
        BATCH_SIZE,
        "model.layers.0.linear_attn".to_string(),
    );

    assert_eq!(qkv_state.shape, vec![SEQUENCE_LENGTH, BATCH_SIZE, QKV_COLS]);
    assert_eq!(z_state.shape, vec![input_rows, VALUE_DIM]);
    assert_eq!(beta_state.shape, vec![input_rows, HEAD_COLS]);
    assert_eq!(g_state.shape, vec![input_rows, HEAD_COLS]);

    let operator = take_single_f32_operator();
    assert!(matches!(operator, Operator::MatMulProj(_)));
}

#[test]
fn test_causal_conv_silu_enqueue() {
    init_f32_test();

    // qkv cache from matmul_proj, convolved in place; rolling state holds
    // the previous kernel_size - 1 = 3 tokens per batch and channel.
    // matmul_proj 的 qkv 缓存，原地卷积；滚动状态保存每 batch 每通道
    // 前 kernel_size - 1 = 3 个 token。
    const KERNEL_SIZE: usize = 4;
    let qkv_state = Tensor::<f32>::from_mem_pool(
        vec![SEQUENCE_LENGTH, BATCH_SIZE, QKV_COLS],
        "model.layers.0.linear_attn.qkv_proj.output".to_string(),
    );
    let weight = Tensor::<f32>::from_mem_pool(
        vec![QKV_COLS, KERNEL_SIZE],
        "model.layers.0.linear_attn.conv1d.weight".to_string(),
    );
    let state = Tensor::<f32>::from_mem_pool(
        vec![BATCH_SIZE, QKV_COLS, KERNEL_SIZE - 1],
        "model.layers.0.linear_attn.conv_state".to_string(),
    );

    qkv_state.causal_conv_silu(
        &weight,
        &state,
        KEY_DIM,
        HEAD_K_DIM,
        SEQUENCE_LENGTH,
        BATCH_SIZE,
    );

    let operator = take_single_f32_operator();
    assert!(matches!(operator, Operator::CausalConv1dSilu(_)));
}

#[test]
fn test_recurrent_gated_delta_rule_enqueue() {
    init_f32_test();

    // num_k_heads = 1, head_k_dim = 2 -> key_dim = 2;
    // num_v_heads = 2, head_v_dim = 2 -> value_dim = 4;
    // conv_dim = 2 * key_dim + value_dim = 8.
    // num_k_heads = 1、head_k_dim = 2 -> key_dim = 2；
    // num_v_heads = 2、head_v_dim = 2 -> value_dim = 4；
    // conv_dim = 2 * key_dim + value_dim = 8。
    const RECURRENT_KEY_DIM: usize = 2;
    const RECURRENT_VALUE_DIM: usize = 4;
    const RECURRENT_CONV_DIM: usize = 2 * RECURRENT_KEY_DIM + RECURRENT_VALUE_DIM;
    const NUM_V_HEADS: usize = 2;
    const HEAD_V_DIM: usize = 2;

    let qkv_state = Tensor::<f32>::from_mem_pool(
        vec![SEQUENCE_LENGTH, BATCH_SIZE, RECURRENT_CONV_DIM],
        "model.layers.0.linear_attn.qkv_proj.output".to_string(),
    );
    let input_rows = SEQUENCE_LENGTH * BATCH_SIZE;
    let g_state = Tensor::<f32>::from_mem_pool(
        vec![input_rows, NUM_V_HEADS],
        "model.layers.0.linear_attn.g_proj.output".to_string(),
    );
    let beta_state = Tensor::<f32>::from_mem_pool(
        vec![input_rows, NUM_V_HEADS],
        "model.layers.0.linear_attn.beta_proj.output".to_string(),
    );
    let state = Tensor::<f32>::from_mem_pool(
        vec![BATCH_SIZE, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM],
        "model.layers.0.linear_attn.recurrent_state".to_string(),
    );

    let output = qkv_state.recurrent_gated_delta_rule(
        &g_state,
        &beta_state,
        &state,
        RECURRENT_KEY_DIM,
        SEQUENCE_LENGTH,
        BATCH_SIZE,
        "model.layers.0.linear_attn".to_string(),
    );

    assert_eq!(
        output.shape,
        vec![SEQUENCE_LENGTH, BATCH_SIZE, RECURRENT_VALUE_DIM]
    );

    let operator = take_single_f32_operator();
    assert!(matches!(operator, Operator::RecurrentGatedDeltaRule(_)));
}
