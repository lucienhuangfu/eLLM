# Implementation Plan: Sparse MoE Refactor

## Overview

将 `src/transformer` 从 Qwen3-MoE 专用路径提升为可扩展的通用 decoder 路径。重构分五个增量步骤，每步完成后运行 `cargo test --workspace` 验证回归安全。

## Tasks

- [x] 1. 将 `FfnKind::SparseMoe` 变体补充 `router_scoring` 和 `use_routing_bias` 字段
  - 在 `src/transformer/config.rs` 中，为 `FfnKind::SparseMoe` 添加 `router_scoring: RouterScoringKind` 和 `use_routing_bias: bool` 两个字段
  - 更新 `resolve_ffn_kind` 函数，从 `HfConfig` 中读取这两个值并填入 `SparseMoe` 变体（`router_scoring` 通过现有的 `resolve_router_scoring` 函数解析，`use_routing_bias` 通过现有逻辑解析）
  - 更新 `decoder_layer.rs` 中 `FfnBlock::SparseMoe` 的构造代码，从 `FfnKind::SparseMoe` 变体中读取 `router_scoring` 和 `use_routing_bias`，而不是从 `config` 顶层读取
  - 删除 `Config`（`ResolvedConfig`）顶层的 `router_scoring: RouterScoringKind` 和 `use_routing_bias: bool` 字段（这两个字段现在只存在于 `FfnKind::SparseMoe` 变体中）
  - _Requirements: 2.3, 4.4, 6.2_

  - [ ]* 1.1 为 `config::tests` 添加属性测试：验证 FfnKind 优先级规则
    - **Property 2: FfnKind priority rule**
    - 在 `Cargo.toml` 中添加 `proptest` dev-dependency（`proptest = "1"`）
    - 在 `config::tests` 中添加 `prop_ffn_kind_priority_rule`：对任意 `layer_idx`、`num_experts`、`mlp_only_layers` 组合，验证生成的 `FfnKind` 遵循优先级规则
    - **Validates: Requirements 2.5, 2.6**

- [x] 2. 清理 `ResolvedConfig` 顶层的 HF 专属字段
  - 从 `Config`（`ResolvedConfig`）结构体中移除以下字段：`model_type`、`mlp_only_layers`、`num_experts`、`num_experts_per_tok`、`moe_intermediate_size`、`norm_topk_prob`、`decoder_sparse_step`、`router_aux_loss_coef`、`output_router_logits`、`shared_experts_intermediate_size`（这些字段的信息已编码进 `layers: Vec<LayerPlan>`）
  - 保留运行时仍需要的字段：`family`、`vocab_size`、`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`head_dim`、`max_position_embeddings`、`rms_norm_eps`、`rope_theta`、`rotary_dim`、`tie_word_embeddings`、`layers`、`qkv_bias`、`use_qk_norm`、`rope_scaling`、`eos_token_id`、`max_window_layers`、`use_sliding_window`、`sliding_window`、`intermediate_size`（dense 层用）
  - 修复所有因字段删除导致的编译错误（主要在 `model.rs`、`attention.rs`、`decoder_layer.rs` 中）
  - 确保 `HfConfig` 结构体保持私有（`struct HfConfig`，不加 `pub`）
  - _Requirements: 1.1, 1.4, 1.5, 1.6_

  - [ ]* 2.1 为 `config::tests` 添加属性测试：验证 layers 长度等于 num_hidden_layers
    - **Property 1: layers length equals num_hidden_layers**
    - 在 `config::tests` 中添加 `prop_layers_len_equals_num_hidden_layers`：构造最小合法 `HfConfig`（可变 `num_hidden_layers`），验证 `ResolvedConfig.layers.len() == num_hidden_layers`
    - **Validates: Requirements 2.1**

- [x] 3. 修复 `names.rs`：`router_bias` 按 `use_routing_bias` 条件生成
  - 在 `layer_tensor_names` 函数中，`FfnKind::SparseMoe` 分支读取 `use_routing_bias` 字段
  - 当 `use_routing_bias == true` 时，`router_bias = Some(format!("{}.e_score_correction_bias", ffn_scope))`
  - 当 `use_routing_bias == false` 时，`router_bias = None`
  - 在 `ModelTensorNames` 结构体中添加 `norm_weight: String` 字段，并在 `model_tensor_names` 函数中填充（值为 `"model.norm.weight"`）
  - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6_

  - [ ]* 3.1 为 `names::tests` 添加属性测试：验证 Qwen SparseMoe 层的 tensor 名称格式
    - **Property 3: Qwen SparseMoe tensor name format**
    - 新建 `src/transformer/names.rs` 中的 `#[cfg(test)] mod tests`（若不存在）
    - 添加 `prop_qwen_sparse_moe_tensor_names`：对任意层索引 `i`（0..=63），构造 Qwen SparseMoe 配置，验证 `router_gate`、`experts_gate_proj`、`experts_up_proj`、`experts_down_proj` 的格式
    - **Validates: Requirements 3.5**

  - [ ]* 3.2 为 `names::tests` 添加单元测试：验证 `model_tensor_names` 返回正确字段
    - 添加 `test_model_tensor_names`：调用 `model_tensor_names`，验证 `token_embedding`、`lm_head`、`norm_weight` 字段值
    - _Requirements: 3.6_

- [ ] 4. 清理 `SparseMoe` 组件：移除 `format!` fallback 和 `println!`
  - 在 `src/transformer/sparse_moe/layer.rs` 中，将 `SparseMoe::new` 里的 `router_bias` 构造改为：当 `use_routing_bias == true` 时，使用 `names.router_bias.expect("use_routing_bias is true but SparseMoeTensorNames.router_bias is None")`，移除 `unwrap_or_else(|| format!(...))` fallback
  - 将 `SparseMoe::forward` 中的 `println!("Entering SparseMoe forward: {}", tensor_name)` 替换为 `#[cfg(debug_assertions)] eprintln!("Entering SparseMoe forward: {}", tensor_name)`
  - _Requirements: 6.1, 6.5_

  - [ ]* 4.1 运行现有 `sparse_moe::tests` 验证行为不变
    - 运行 `cargo test -- sparse_moe::tests`，确认全部 4 个测试通过
    - 验证 `test_sparse_moe_queue_structure`、`test_sparse_moe_sigmoid_queue_structure`、`test_sparse_moe_zero_weights_output_equals_residual_bits`、`test_sparse_moe_single_thread_equals_multi_thread_bits` 均通过
    - _Requirements: 6.4, 8.2_

- [~] 5. 修复 `DenseMlp::forward` 签名：移除 `_tensor_name` 参数
  - 在 `src/transformer/dense_mlp.rs` 中，将 `pub fn forward(&self, hidden_states: &Tensor<T>, residual: &Tensor<T>, _tensor_name: String) -> Tensor<T>` 改为 `pub fn forward(&self, hidden_states: &Tensor<T>, residual: &Tensor<T>) -> Tensor<T>`
  - 在 `src/transformer/decoder_layer.rs` 中，更新 `FfnBlock::Dense(dense_mlp)` 分支的调用，移除 `format!(...)` 参数
  - _Requirements: 5.2, 5.3_

- [~] 6. Checkpoint — 运行全量测试，确保所有测试通过
  - 运行 `cargo test --workspace`，确保所有测试通过，包括 `sparse_moe::tests`、`decoder_layer::test`、`model::test`、`config::tests`、`attention::test`
  - 如有失败，修复后再继续
  - _Requirements: 7.5, 8.1_

- [ ] 7. 补充 `config::tests` 中的回归和错误路径测试
  - 添加 `test_load_all_known_configs`：依次加载 `models/Qwen3-Coder-30B-A3B-Instruct/config.json`、`models/Llama-2-7b-hf/config.json`、`models/Llama-2-70b-hf/config.json`、`models/MiniMax-M2.5/config.json`，验证每个均返回 `Ok(_)`
  - 添加 `test_qwen3_layer_plan`：加载 Qwen3-Coder-30B 配置，验证每层 `FfnKind` 均为 `SparseMoe`，且 `router_scoring == RouterScoringKind::Softmax`，`use_routing_bias == false`
  - 添加 `test_missing_field_returns_err`：传入缺少 `hidden_size` 字段的最小 JSON，验证 `load_from_file` 返回 `Err`（注：`serde_json` 对有 `#[serde(default)]` 的字段不会报错，此测试针对真正必要的字段）
  - _Requirements: 1.7, 7.1, 8.4, 8.5_

  - [ ]* 7.1 为 `config::tests` 添加属性测试：验证所有已知配置文件均可成功加载
    - **Property 5: all known configs load successfully**
    - 添加 `test_all_known_configs_load`（参数化测试，遍历 4 个已知配置文件路径）
    - **Validates: Requirements 8.4**

- [ ] 8. 补充 `decoder_layer::test` 中的属性测试
  - 在 `src/transformer/decoder_layer.rs` 的 `#[cfg(test)] mod test` 中添加属性测试 `prop_zero_weight_layer_output_equals_residual`
  - 使用 `proptest` 生成不同的 `batch_size`（1..=8）和输入值，构造零权重 `DecoderLayer`（层索引 = 1），运行 `forward`，验证输出 tensor 的每个 `f16::to_bits()` 值等于输入 residual 的对应值
  - _Requirements: 4.6, 8.3_

  - [ ]* 8.1 运行属性测试验证零权重层行为
    - **Property 4: zero-weight layer output equals residual**
    - 运行 `cargo test -- decoder_layer::test::prop_zero_weight_layer_output_equals_residual`
    - **Validates: Requirements 4.6, 8.3**

- [~] 9. Final Checkpoint — 运行全量测试，确认重构完成
  - 运行 `cargo test --workspace`，确保所有测试通过
  - 确认无任何 `println!` 在 release 路径中（`SparseMoe::forward` 中已改为 `#[cfg(debug_assertions)] eprintln!`）
  - 确认 `src/bin/main.rs` 中的 `Config::load_from_file(path)` 调用无需修改即可编译
  - _Requirements: 7.2, 7.3, 7.5, 8.1_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- 重构顺序设计为增量式：每步完成后代码均可编译并通过现有测试
- Task 1 是关键路径：`FfnKind::SparseMoe` 补充字段后，Task 2（清理顶层字段）和 Task 3（names 修复）才能安全进行
- `proptest` crate 需要在 `Cargo.toml` 的 `[dev-dependencies]` 中添加
- 属性测试每个至少运行 100 次迭代（proptest 默认值）
- 每个属性测试注释中包含 `Feature: sparse-moe-refactor, Property N: <property_text>` 标记
