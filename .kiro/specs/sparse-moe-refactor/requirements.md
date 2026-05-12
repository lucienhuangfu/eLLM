# Requirements Document

## Introduction

本次重构的目标是将 `src/transformer` 从"Qwen3-MoE 专用路径"提升为"可扩展到 dense / sparse MoE / hybrid attention 的通用 decoder 路径"，同时尽量少改现有代码。

重构以 `docs/en/transformers/minimal_model_abstraction.md` 为设计蓝图，核心是拆解三处耦合：

1. `config.rs` 直接承载某个具体模型的 JSON 结构
2. `decoder_layer.rs` 写死为 `Attention + SparseMoeBlock`
3. tensor 名称拼接分散在 model / attention / block 构造函数中

通过引入三个最小抽象（`ModelFamily`、`LayerPlan`、`TensorNames`），使 `sparse_moe` 模块成为通用 FFN 组件体系的一部分，而不是 Qwen3-MoE 的专属实现。

## Glossary

- **HfConfig**: 忠实映射 HuggingFace 原始 JSON 的配置结构，允许 `Option` 字段
- **ResolvedConfig**: 运行时使用的稳定配置，字段完整，不含 family 特例判断；当前阶段可通过 `pub type Config = ResolvedConfig` 保持调用兼容
- **ModelFamily**: 模型族标识枚举，用于前置分发，避免运行时核心逻辑中出现 `if model_type == ...`
- **LayerPlan**: 每层的结构描述，包含 `AttentionKind` 和 `FfnKind`，在构建阶段前置决定每层的 block 组合
- **FfnKind**: FFN 类型枚举，当前包含 `Dense` 和 `SparseMoe` 两个变体
- **AttentionKind**: Attention 类型枚举，当前包含 `Full`、`SlidingWindow`、`Linear`（占位）
- **FfnBlock**: `DecoderLayer` 持有的 FFN 执行块枚举，包含 `Dense(DenseMlp)` 和 `SparseMoe(SparseMoe)` 两个变体
- **AttentionBlock**: `DecoderLayer` 持有的 Attention 执行块枚举
- **TensorNames / LayerTensorNames / FfnTensorNames**: 描述各组件所需 tensor 名称的结构体，与具体 family 解耦
- **SparseMoe**: `sparse_moe` 模块中的核心结构体，负责 MoE FFN 的前向计算
- **SparseMoeTensorNames**: 描述 SparseMoe 所需 tensor 名称的结构体
- **DenseMlp**: `dense_mlp` 模块中的 dense FFN 结构体，与 `SparseMoe` 并列为 `FfnBlock` 的两个变体
- **RouterScoringKind**: 路由评分方式枚举，包含 `Softmax` 和 `Sigmoid` 两个变体
- **FamilyResolver**: 将 `HfConfig` 转换为 `ResolvedConfig` 的函数集合，按 family 分发

---

## Requirements

### Requirement 1: 将 Config 拆分为 HfConfig 和 ResolvedConfig 两层

**User Story:** As a 运行时开发者, I want config 层只暴露稳定的 ResolvedConfig 给运行时模块, so that 运行时代码不再需要处理 HuggingFace JSON 的 Option 字段和 family 特例。

#### Acceptance Criteria

1. THE Config_Module SHALL 将当前 `Config` 结构体拆分为 `HfConfig`（原始 JSON 映射）和 `ResolvedConfig`（运行时稳定配置）两个独立结构体，两者均定义在 `src/transformer/config.rs` 中。
2. WHEN 从文件加载配置时，THE Config_Module SHALL 先将 JSON 反序列化为 `HfConfig`，再通过 family-specific resolver 转换为 `ResolvedConfig`，整个过程在 `Config::load_from_file` 内部完成。
3. THE Config_Module SHALL 通过 `pub type Config = ResolvedConfig` 保持现有调用接口兼容，使 `src/bin/main.rs` 中的 `Config::load_from_file(path)` 调用无需修改即可编译通过。
4. THE ResolvedConfig SHALL 包含完整的运行时字段（所有字段均为非 `Option` 类型），至少包括 `family: ModelFamily`、`vocab_size: usize`、`hidden_size: usize`、`num_hidden_layers: usize`、`num_attention_heads: usize`、`num_key_value_heads: usize`、`head_dim: usize`、`max_position_embeddings: usize`、`rms_norm_eps: f64`、`rope_theta: f64`、`tie_word_embeddings: bool`、`layers: Vec<LayerPlan>`。
5. THE HfConfig SHALL 仅包含 `pub` 字段（允许 `Option<T>` 类型）且不实现任何运行时方法；`HfConfig` 不得在 `config.rs` 模块外部被直接引用。
6. WHEN family 特例判断（如 `use_routing_bias` 默认值、`scoring_func` 字符串解析）发生时，THE Config_Module SHALL 仅在 `HfConfig -> ResolvedConfig` 的转换函数中处理；config 模块之外的所有模块（`attention.rs`、`layer.rs`、`model.rs` 等）不得包含任何 `model_type` 字符串比较或 family 条件分支。
7. IF 加载的 JSON 缺少 `ResolvedConfig` 所需的必要字段，THEN THE Config_Module SHALL 返回包含缺失字段名称的 `Err`（如 `"missing field: hidden_size"`），而不是 panic 或返回不含字段名的通用错误。

---

### Requirement 2: 在 ResolvedConfig 中引入 LayerPlan 前置层结构

**User Story:** As a 模型构建者, I want 每层的 attention 类型和 FFN 类型在构建阶段就已确定, so that DecoderLayer 不再需要在运行时推断层结构，也不需要包含 family 特例逻辑。

#### Acceptance Criteria

1. THE Config_Module SHALL 在 `ResolvedConfig` 中包含 `layers: Vec<LayerPlan>` 字段，`layers.len()` 必须严格等于 `num_hidden_layers`；若两者不一致，`load_from_file` 应返回 `Err`。
2. THE LayerPlan SHALL 包含 `attention: AttentionKind` 和 `ffn: FfnKind` 两个字段，均为非 `Option` 类型。
3. THE FfnKind SHALL 包含 `Dense { intermediate_size: usize }` 和 `SparseMoe { intermediate_size: usize, num_experts: usize, num_experts_per_tok: usize, norm_topk_prob: bool, router_scoring: RouterScoringKind, use_routing_bias: bool }` 两个变体；所有路由相关参数随 `SparseMoe` 变体携带，不得保留在 `ResolvedConfig` 顶层。
4. THE AttentionKind SHALL 包含 `Full`、`SlidingWindow { window_size: usize }`、`Linear` 三个变体；WHEN `Linear` 变体的 `forward` 被调用时，THE AttentionBlock SHALL panic 并输出消息 `"Linear attention is not implemented in this phase"`。
5. WHEN 构建 `ResolvedConfig` 时，THE Config_Module SHALL 按以下优先级为每一层确定 `FfnKind`：(1) 若层索引在 `mlp_only_layers` 列表中，则为 `Dense`；(2) 否则若 `num_experts > 0`，则为 `SparseMoe`；(3) 否则为 `Dense`。
6. WHEN 构建 Qwen3-MoE 的 `ResolvedConfig` 时，THE Config_Module SHALL 对 `mlp_only_layers` 为空列表的情况正确处理（即所有层均为 `SparseMoe`），不得将空列表误判为"所有层均为 Dense"。
7. WHEN 加载 `models/Qwen3-Coder-30B-A3B-Instruct/config.json` 时，THE Config_Module SHALL 生成的 `layers` 序列中每个 `LayerPlan` 的 `ffn` 变体类型（`Dense` 或 `SparseMoe`）与重构前 `decoder_layer.rs` 中的运行时判断结果完全一致（逐层比对）。

---

### Requirement 3: 将 tensor 名称拼接集中到 names.rs

**User Story:** As a 模型适配开发者, I want tensor 名称的生成逻辑集中在 names.rs 中, so that 添加新模型族时只需修改 names.rs，不需要改动 attention、sparse_moe、model 等核心组件。

#### Acceptance Criteria

1. THE Names_Module SHALL 提供 `layer_tensor_names(config: &Config, layer_idx: usize) -> LayerTensorNames` 函数（`Config` 为 `ResolvedConfig` 的类型别名），为每一层生成完整的 tensor 名称描述；该函数是 `names.rs` 中唯一允许拼接 tensor 名称字符串的位置。
2. THE LayerTensorNames SHALL 包含 `scope: String`、`attention: AttentionTensorNames`、`ffn: FfnTensorNames` 字段，其中 `FfnTensorNames` 为枚举类型，与 `FfnKind` 变体一一对应。
3. THE FfnTensorNames SHALL 为 `Dense` 变体包含 `DenseMlpTensorNames { scope: String, gate_proj: String, up_proj: String, down_proj: String }` 内层结构；为 `SparseMoe` 变体包含 `SparseMoeTensorNames { scope: String, router_gate: String, router_bias: Option<String>, experts_gate_proj: String, experts_up_proj: String, experts_down_proj: String }` 内层结构。
4. WHEN `SparseMoe::new`、`Attention::new`、`Model::new` 被调用时，THE Components SHALL 不包含任何 `format!` 调用用于拼接 tensor 名称；所有 tensor 名称字符串必须来自调用方传入的 `*TensorNames` 结构体字段；WHEN `use_routing_bias` 为 `true` 时，调用方必须在 `SparseMoeTensorNames.router_bias` 中提供 `Some(name)`，`SparseMoe::new` 不得在内部生成 fallback 名称。
5. WHEN `layer_tensor_names` 以 `ModelFamily::Qwen`（对应 `model_type = "qwen3_moe"`）和层索引 `i` 被调用时，THE Names_Module SHALL 生成的 `SparseMoeTensorNames` 字段值与当前 `layer.rs` 中硬编码的字符串完全一致，例如 `router_gate = "model.layers.{i}.mlp.gate.weight"`、`experts_gate_proj = "model.layers.{i}.mlp.experts.{j}.gate_proj.weight"`（`j` 为专家索引占位符格式）。
6. THE Names_Module SHALL 提供 `model_tensor_names(config: &Config) -> ModelTensorNames` 函数，返回包含 `embed_tokens: String`、`lm_head: String`、`norm_weight: String` 字段的 `ModelTensorNames` 结构体；WHEN `tie_word_embeddings` 为 `true` 时，`lm_head` 字段值应与 `embed_tokens` 相同。

---

### Requirement 4: 将 DecoderLayer 改为基于 AttentionBlock 和 FfnBlock 的枚举分发

**User Story:** As a 运行时开发者, I want DecoderLayer 通过枚举持有具体的 attention 和 FFN 块, so that dense 和 sparse MoE 两条路径可以在同一个 DecoderLayer 框架下并存，不需要 trait object。

#### Acceptance Criteria

1. THE DecoderLayer SHALL 持有 `attention: AttentionBlock<T>` 和 `ffn: FfnBlock<T>` 两个枚举字段，替代当前对 `Attention<T>` 和 `SparseMoe<T>` 的直接持有；`AttentionBlock` 和 `FfnBlock` 均定义在 `src/transformer/decoder_layer.rs` 或独立的 `blocks.rs` 中。
2. THE FfnBlock SHALL 包含 `Dense(DenseMlp<T>)` 和 `SparseMoe(SparseMoe<T>)` 两个变体，不得包含其他变体。
3. THE AttentionBlock SHALL 包含 `Full(Attention<T>)` 和 `SlidingWindow(Attention<T>)` 两个变体；WHEN `AttentionBlock::Linear` 的 `forward` 被调用时，THE AttentionBlock SHALL panic 并输出消息 `"Linear attention is not implemented in this phase"`（`Linear` 变体可作为占位符存在，但不得在 Qwen3-MoE 路径中被构建）。
4. WHEN `DecoderLayer::new` 被调用时，THE DecoderLayer SHALL 仅根据传入的 `LayerPlan.ffn` 和 `LayerPlan.attention` 字段选择枚举变体；构造函数中不得出现任何 `model_type` 字符串比较、`family` 条件分支或其他 family 特例判断。
5. THE DecoderLayer::forward SHALL 通过 `match self.ffn { FfnBlock::Dense(ref mlp) => ..., FfnBlock::SparseMoe(ref moe) => ... }` 对 `FfnBlock` 进行静态分发，不得使用 `dyn Trait` 动态分发；`AttentionBlock` 同理。
6. WHEN `DecoderLayer::forward` 以零权重初始化的 `DecoderLayer`（层索引 ≥ 1）被调用时，THE DecoderLayer SHALL 生成与重构前位级别相同的输出 tensor（即输出等于输入 residual，`f16::to_bits()` 逐元素相等）。

---

### Requirement 5: 将 DenseMlp 恢复为正式的一等公民 FFN 组件

**User Story:** As a 模型开发者, I want DenseMlp 与 SparseMoe 并列作为 FfnBlock 的两个变体, so that dense MLP 模型（如 Llama）可以通过相同的 DecoderLayer 框架接入，不需要特殊处理。

#### Acceptance Criteria

1. THE DenseMlp SHALL 接受 `hidden_size: usize`、`intermediate_size: usize`、`names: DenseMlpTensorNames`、`ctx: Rc<TensorCtx<T>>` 作为构造参数，不得接受 `Config` 对象或任何包含 family 信息的参数。
2. THE DenseMlp::forward SHALL 接受 `hidden_states: &Tensor<T>` 和 `residual: &Tensor<T>` 两个参数，返回 `Tensor<T>`；不得包含 `tensor_name: String` 调试参数（与 `SparseMoe::forward` 的两参数契约对称，dense MLP 无路由路径，不需要调试追踪参数）。
3. THE DenseMlp SHALL 通过 `FfnBlock::Dense` 变体在 `DecoderLayer` 中被使用；`DecoderLayer` 不得在 `FfnBlock` 枚举之外直接持有 `DenseMlp<T>` 字段。
4. WHEN `FfnKind::Dense` 层被构建时，THE DecoderLayer SHALL 调用 `layer_tensor_names(config, layer_idx)` 获取 `FfnTensorNames::Dense(names)`，并将 `names` 传入 `DenseMlp::new`；WHEN `FfnKind::SparseMoe` 层被构建但代码路径进入 `Dense` 分支时（或反之），THE DecoderLayer SHALL 以 `unreachable!()` 处理该不变量违反。

---

### Requirement 6: SparseMoe 组件接受 SparseMoeTensorNames 而非散列参数

**User Story:** As a 模型适配开发者, I want SparseMoe::new 只接受 SparseMoeTensorNames 和必要的数值参数, so that 添加新模型族时只需提供不同的 SparseMoeTensorNames，不需要修改 SparseMoe 内部逻辑。

#### Acceptance Criteria

1. THE SparseMoe SHALL 通过 `SparseMoeTensorNames` 结构体接收所有 tensor 名称；`SparseMoe::new` 内部不得包含任何 `format!` 调用用于拼接 tensor 名称字符串，包括当前存在的 `format!("{}.e_score_correction_bias", scope_name)` fallback；WHEN `use_routing_bias` 为 `true` 时，调用方必须在 `SparseMoeTensorNames.router_bias` 中提供 `Some(name)`，否则 `SparseMoe::new` 应返回 `Err` 或 panic 并附带说明。
2. THE SparseMoe::new SHALL 接受以下参数：`hidden_size: usize`、`intermediate_size: usize`、`num_experts: usize`、`num_topk: usize`、`norm_topk_prob: bool`、`router_scoring: RouterScoringKind`、`use_routing_bias: bool`、`names: SparseMoeTensorNames`、`ctx: Rc<TensorCtx<T>>`；不得接受 `Config` 对象或任何包含 family 信息的参数。
3. THE SparseMoeTensorNames SHALL 包含 `scope: String`、`router_gate: String`、`router_bias: Option<String>`、`experts_gate_proj: String`、`experts_up_proj: String`、`experts_down_proj: String` 字段，所有字段均为 `pub`。
4. WHEN `cargo test -- sparse_moe::tests` 运行时，THE SparseMoe SHALL 通过 `tests::test_sparse_moe_forward`、`tests::test_sparse_moe_routing`、`tests::test_sparse_moe_norm_topk`、`tests::test_sparse_moe_sigmoid_router` 全部四个测试，且每个测试的输出 tensor 与重构前的输出 tensor 在 `f16::to_bits()` 逐元素比较下完全相等。
5. WHEN `SparseMoe::forward` 被调用时，THE SparseMoe SHALL 不向 stdout 输出任何内容；原有的 `println!` 调试输出 SHALL 被替换为 `#[cfg(debug_assertions)] eprintln!(...)`，在 release 构建中不产生任何 I/O。

---

### Requirement 7: 第一阶段仅接通 Qwen3-MoE，证明抽象有效性

**User Story:** As a 项目维护者, I want 重构后的代码在第一阶段仍然只运行 Qwen3-MoE 路径, so that 可以在不引入新模型风险的前提下验证抽象层的正确性。

#### Acceptance Criteria

1. WHEN 加载 `models/Qwen3-Coder-30B-A3B-Instruct/config.json` 时，THE Config_Module SHALL 生成的 `ResolvedConfig` 在以下字段上与重构前等价：`family`、`hidden_size`、`num_hidden_layers`、`head_dim`、`rms_norm_eps`、`rope_theta`、`num_attention_heads`、`num_key_value_heads`；且 `layers` 序列中每个 `LayerPlan` 的 `ffn` 变体类型（`Dense` 或 `SparseMoe`）及其 payload 字段值与重构前逐层一致。
2. IF 以 `batch_size=3, sequence_length=128, token_ids=all_zeros` 为输入运行 `model::test::test_model_forward` 测试，THEN THE Model SHALL 在重构后产生与重构前位级别相同的输出 logits（`f16::to_bits()` 逐元素相等）。
3. THE Config_Module SHALL 保持 `Config::load_from_file(path: P) -> Result<Config, _>` 函数签名不变，使 `src/bin/main.rs` 中的现有调用代码无需任何修改即可编译通过。
4. IF Llama 模型族支持被添加，THEN THE Names_Module SHALL 能够通过新增 family resolver 函数支持 Llama 命名规则，而无需修改 `SparseMoe`、`DenseMlp`、`Attention`、`DecoderLayer` 四个核心组件中的任何代码。
5. THE Refactored_Codebase SHALL 通过 `cargo test --workspace` 的全部测试用例，包括 `sparse_moe::tests`、`decoder_layer::test`、`model::test`、`config::tests` 中的全部测试用例。

---

### Requirement 8: 重构过程保持行为等价性（回归安全网）

**User Story:** As a 质量保证工程师, I want 重构的每个步骤都有可验证的行为等价性保证, so that 重构不会引入静默的数值错误或接口破坏。

#### Acceptance Criteria

1. WHEN `cargo test --workspace` 运行时，THE Refactored_Codebase SHALL 使所有测试用例保持与重构前相同的 pass/fail 状态；不得有任何原本通过的测试在重构后失败。
2. WHEN `SparseMoe::forward` 以相同输入被调用时，THE SparseMoe SHALL 生成与重构前相同的 operator enum 变体序列（类型和顺序不变），以 `tests.rs` 中现有的 operator queue 断言为基准。
3. WHEN `DecoderLayer::forward` 以零权重初始化的层（层索引 ≥ 1）和任意输入 tensor 被调用时，THE DecoderLayer SHALL 生成与重构前位级别相同的输出 tensor（`f16::to_bits()` 逐元素相等）。
4. WHEN `Config::load_from_file` 分别以 `models/Qwen3-Coder-30B-A3B-Instruct/config.json`、`models/Llama-2-7b-hf/config.json`、`models/Llama-2-70b-hf/config.json`、`models/MiniMax-M2.5/config.json` 为参数被调用时，THE Config_Module SHALL 对每个文件返回 `Ok(_)`，不得 panic 或返回 `Err`。
5. IF 重构引入了新的公开 API（新的 `pub fn`、`pub struct`、`pub enum`），THEN THE New_API SHALL 包含对应的单元测试，每个新 API 至少覆盖 1 个正常路径测试和 1 个错误路径测试（如无效输入、缺失字段等）。
