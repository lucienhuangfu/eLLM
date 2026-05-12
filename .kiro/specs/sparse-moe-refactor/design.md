# Design Document: Sparse MoE Refactor

## Overview

本次重构将 `src/transformer` 从"Qwen3-MoE 专用路径"提升为"可扩展到 dense / sparse MoE / hybrid attention 的通用 decoder 路径"。设计蓝图来自 `docs/en/transformers/minimal_model_abstraction.md`，核心原则是：**family-first、layer-plan-first、names-outside-components**。

重构分三个关注点：

1. **Config 层**：将现有单一 `Config` 拆分为 `HfConfig`（原始 JSON 映射）和 `ResolvedConfig`（运行时稳定配置），通过 `pub type Config = ResolvedConfig` 保持调用兼容。
2. **Names 层**：将 tensor 名称拼接集中到 `names.rs`，核心组件（`SparseMoe`、`DenseMlp`、`Attention`）不再包含任何 `format!` 调用。
3. **Layer 层**：`DecoderLayer` 通过 `AttentionBlock<T>` 和 `FfnBlock<T>` 枚举持有具体块，构造时仅依赖 `LayerPlan`，不含 family 特例逻辑。

### 当前状态与目标差距

通过阅读现有代码，识别出以下具体差距：

| 位置 | 当前状态 | 目标状态 |
|------|---------|---------|
| `config.rs` / `FfnKind::SparseMoe` | 缺少 `router_scoring` 和 `use_routing_bias` 字段 | 携带所有路由参数 |
| `config.rs` / `Config` | 顶层仍有 `router_scoring`、`use_routing_bias`、`model_type` 等 HF 字段 | 这些字段移入 `HfConfig`，不暴露在 `ResolvedConfig` 顶层 |
| `names.rs` / `SparseMoeTensorNames` | `router_bias` 始终为 `Some(...)`，不管 `use_routing_bias` | 由调用方（`layer_tensor_names`）根据 `use_routing_bias` 决定是否提供 |
| `sparse_moe/layer.rs` | `SparseMoe::new` 内有 `format!` fallback；`forward` 有 `println!` | 移除 `format!` fallback；`println!` 改为 `#[cfg(debug_assertions)] eprintln!` |
| `dense_mlp.rs` / `DenseMlp::forward` | 有 `_tensor_name: String` 参数 | 移除该参数，与 requirements 中的两参数契约对称 |
| `config.rs` / `HfConfig` | 是私有结构体，但 `Config` 顶层仍暴露 HF 字段 | `HfConfig` 完全私有，`ResolvedConfig` 只含运行时字段 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    src/transformer/                          │
│                                                             │
│  config.rs                                                  │
│  ┌──────────────┐    resolve    ┌──────────────────────┐   │
│  │  HfConfig    │ ──────────►  │  ResolvedConfig       │   │
│  │  (private)   │              │  (pub type Config=...) │   │
│  └──────────────┘              │  + layers: Vec<LayerPlan>│  │
│                                └──────────────────────┘   │
│                                                             │
│  names.rs                                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  layer_tensor_names(config, layer_idx) -> LayerTensorNames│
│  │  model_tensor_names(config) -> ModelTensorNames       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  decoder_layer.rs                                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DecoderLayer<T>                                      │  │
│  │    attention: AttentionBlock<T>  ──► Attention<T>     │  │
│  │    ffn: FfnBlock<T>  ──► DenseMlp<T> | SparseMoe<T>  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

数据流：

```
JSON file
  │
  ▼
HfConfig (serde_json)
  │  resolve_family() + resolve_ffn_kind() + resolve_attention_kind()
  ▼
ResolvedConfig (= Config)
  │  layers: Vec<LayerPlan>
  ▼
Model::new()
  │  layer_tensor_names(config, i) for each layer
  ▼
DecoderLayer::new(config, layer_idx, names)
  │  match LayerPlan.ffn / LayerPlan.attention
  ▼
AttentionBlock<T> + FfnBlock<T>
```

---

## Components and Interfaces

### 1. `config.rs` — Config 层

#### `HfConfig`（私有）

```rust
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct HfConfig {
    // 原始 JSON 字段，允许 Option<T>
    model_type: String,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    intermediate_size: Option<usize>,
    moe_intermediate_size: Option<usize>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    norm_topk_prob: bool,
    mlp_only_layers: Vec<usize>,
    scoring_func: Option<String>,
    use_routing_bias: Option<bool>,
    rope_theta: Option<usize>,
    rms_norm_eps: f32,
    max_position_embeddings: usize,
    tie_word_embeddings: bool,
    vocab_size: usize,
    use_sliding_window: bool,
    sliding_window: Option<usize>,
    max_window_layers: Option<usize>,
    layer_types: Option<Vec<String>>,
    decoder_sparse_step: usize,
    rotary_dim: Option<usize>,
    rope_scaling: Option<HashMap<String, Value>>,
    // ... 其他 HF 字段
}
```

#### `ResolvedConfig`（公开，通过 `pub type Config = ResolvedConfig`）

```rust
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub family: ModelFamily,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: usize,
    pub rotary_dim: usize,
    pub tie_word_embeddings: bool,
    pub layers: Vec<LayerPlan>,
    // 保留运行时需要的其他字段（qkv_bias, use_qk_norm 等）
    pub qkv_bias: bool,
    pub use_qk_norm: bool,
    pub rope_scaling: Option<HashMap<String, Value>>,
    pub eos_token_id: usize,
}

pub type Config = ResolvedConfig;
```

**关键变化**：`router_scoring`、`use_routing_bias`、`model_type`、`mlp_only_layers`、`num_experts`、`num_experts_per_tok`、`moe_intermediate_size` 等字段**不再**出现在 `ResolvedConfig` 顶层。这些信息已经编码进 `layers: Vec<LayerPlan>` 中。

#### `FfnKind`（更新）

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FfnKind {
    Dense {
        intermediate_size: usize,
    },
    SparseMoe {
        intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        router_scoring: RouterScoringKind,   // 新增
        use_routing_bias: bool,              // 新增
    },
}
```

#### `resolve_ffn_kind`（更新）

```rust
fn resolve_ffn_kind(raw: &HfConfig, layer_idx: usize, ...) -> FfnKind {
    let num_experts = raw.num_experts.unwrap_or(0);
    if raw.mlp_only_layers.contains(&layer_idx) || num_experts == 0 {
        return FfnKind::Dense { intermediate_size: ... };
    }
    if (layer_idx + 1) % decoder_sparse_step == 0 {
        return FfnKind::SparseMoe {
            intermediate_size: moe_intermediate_size,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob,
            router_scoring,    // 从 HfConfig 解析
            use_routing_bias,  // 从 HfConfig 解析
        };
    }
    FfnKind::Dense { intermediate_size: ... }
}
```

### 2. `names.rs` — Names 层

`names.rs` 已经有正确的结构体定义。需要修改的是 `layer_tensor_names` 中 `SparseMoeTensorNames` 的 `router_bias` 字段：

```rust
pub fn layer_tensor_names(config: &Config, layer_idx: usize) -> LayerTensorNames {
    // ...
    let ffn = match &config.layers[layer_idx].ffn {
        FfnKind::SparseMoe { use_routing_bias, .. } => {
            FfnTensorNames::SparseMoe(SparseMoeTensorNames {
                scope: ffn_scope.clone(),
                router_gate: format!("{}.gate.weight", ffn_scope),
                // 仅当 use_routing_bias 为 true 时提供 Some(name)
                router_bias: if *use_routing_bias {
                    Some(format!("{}.e_score_correction_bias", ffn_scope))
                } else {
                    None
                },
                experts_gate_proj: format!("{}.experts.gate_proj.weight", ffn_scope),
                experts_up_proj: format!("{}.experts.up_proj.weight", ffn_scope),
                experts_down_proj: format!("{}.experts.down_proj.weight", ffn_scope),
            })
        }
        // ...
    };
}
```

同时需要补充 `ModelTensorNames` 中的 `norm_weight` 字段（requirements 3.6 要求）：

```rust
pub struct ModelTensorNames {
    pub scope: String,
    pub token_embedding: String,
    pub position_embedding: String,
    pub lm_head: String,
    pub norm_weight: String,  // 新增
}
```

### 3. `sparse_moe/layer.rs` — SparseMoe 组件

**变更 1**：移除 `SparseMoe::new` 中的 `format!` fallback：

```rust
// 当前（需要移除）：
let router_bias = if use_routing_bias {
    Some(ctx.zeros(
        vec![num_experts],
        names.router_bias.clone()
            .unwrap_or_else(|| format!("{}.e_score_correction_bias", scope_name)),  // 移除
    ))
} else {
    None
};

// 目标：
let router_bias = if use_routing_bias {
    let bias_name = names.router_bias
        .expect("use_routing_bias is true but SparseMoeTensorNames.router_bias is None");
    Some(ctx.zeros(vec![num_experts], bias_name))
} else {
    None
};
```

**变更 2**：将 `forward` 中的 `println!` 替换为条件编译的 `eprintln!`：

```rust
// 当前：
println!("Entering SparseMoe forward: {}", tensor_name);

// 目标：
#[cfg(debug_assertions)]
eprintln!("Entering SparseMoe forward: {}", tensor_name);
```

### 4. `dense_mlp.rs` — DenseMlp 组件

**变更**：移除 `forward` 的 `_tensor_name: String` 参数：

```rust
// 当前：
pub fn forward(&self, hidden_states: &Tensor<T>, residual: &Tensor<T>, _tensor_name: String) -> Tensor<T>

// 目标：
pub fn forward(&self, hidden_states: &Tensor<T>, residual: &Tensor<T>) -> Tensor<T>
```

同时更新 `decoder_layer.rs` 中的调用点：

```rust
// 当前：
FfnBlock::Dense(dense_mlp) => dense_mlp.forward(
    &norm_hidden_states,
    &attention_hidden_states,
    format!("{}.attention_hidden3", self.scope_name),
),

// 目标：
FfnBlock::Dense(dense_mlp) => dense_mlp.forward(
    &norm_hidden_states,
    &attention_hidden_states,
),
```

### 5. `decoder_layer.rs` — DecoderLayer 组件

`DecoderLayer` 的枚举结构已经正确实现。需要的变更：

1. 更新 `FfnBlock::SparseMoe` 构造，从 `FfnKind::SparseMoe` 中读取 `router_scoring` 和 `use_routing_bias`（而不是从 `config` 顶层读取）。
2. 更新 `DenseMlp::forward` 调用（移除 `tensor_name` 参数）。

```rust
// 当前（从 config 顶层读取）：
FfnBlock::SparseMoe(SparseMoe::new(
    config.hidden_size,
    *intermediate_size,
    *num_experts,
    *num_experts_per_tok,
    *norm_topk_prob,
    config.router_scoring.clone(),   // ← 从顶层读取
    config.use_routing_bias,         // ← 从顶层读取
    ffn_names,
    ctx.clone(),
))

// 目标（从 FfnKind 变体读取）：
(
    FfnKind::SparseMoe {
        intermediate_size,
        num_experts,
        num_experts_per_tok,
        norm_topk_prob,
        router_scoring,
        use_routing_bias,
    },
    FfnTensorNames::SparseMoe(ffn_names),
) => FfnBlock::SparseMoe(SparseMoe::new(
    config.hidden_size,
    *intermediate_size,
    *num_experts,
    *num_experts_per_tok,
    *norm_topk_prob,
    router_scoring.clone(),   // ← 从 FfnKind 变体读取
    *use_routing_bias,        // ← 从 FfnKind 变体读取
    ffn_names,
    ctx.clone(),
))
```

---

## Data Models

### `LayerPlan`（更新后）

```rust
pub struct LayerPlan {
    pub attention: AttentionKind,
    pub ffn: FfnKind,
}
```

### `FfnKind`（更新后）

```rust
pub enum FfnKind {
    Dense {
        intermediate_size: usize,
    },
    SparseMoe {
        intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        router_scoring: RouterScoringKind,
        use_routing_bias: bool,
    },
}
```

### `AttentionKind`（不变）

```rust
pub enum AttentionKind {
    Full,
    SlidingWindow { window_size: usize },  // 注：当前实现中 SlidingWindow 不携带 window_size，可保持现状
    Linear,
}
```

### `SparseMoeTensorNames`（不变，已正确）

```rust
pub struct SparseMoeTensorNames {
    pub scope: String,
    pub router_gate: String,
    pub router_bias: Option<String>,
    pub experts_gate_proj: String,
    pub experts_up_proj: String,
    pub experts_down_proj: String,
}
```

### `ModelTensorNames`（补充 `norm_weight`）

```rust
pub struct ModelTensorNames {
    pub scope: String,
    pub token_embedding: String,
    pub position_embedding: String,
    pub lm_head: String,
    pub norm_weight: String,
}
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

本重构涉及配置解析、名称生成和层构建三个核心逻辑，其中有若干适合属性测试的普遍性质。

### Property 1: layers 长度与 num_hidden_layers 一致

*For any* 成功加载的配置文件，`config.layers.len()` 必须严格等于 `config.num_hidden_layers`。

**Validates: Requirements 2.1**

### Property 2: FfnKind 优先级规则对所有层成立

*For any* 配置中的每一层索引 `i`，其 `FfnKind` 必须遵循以下优先级：若 `i` 在 `mlp_only_layers` 中则为 `Dense`；否则若 `num_experts > 0` 且满足 `decoder_sparse_step` 条件则为 `SparseMoe`；否则为 `Dense`。

**Validates: Requirements 2.5, 2.6**

### Property 3: Qwen 族 SparseMoe 层的 tensor 名称格式

*For any* 层索引 `i`，当 `config.family == ModelFamily::Qwen` 且该层为 `SparseMoe` 时，`layer_tensor_names(config, i)` 生成的 `SparseMoeTensorNames` 必须满足：
- `router_gate == "model.layers.{i}.mlp.gate.weight"`
- `experts_gate_proj == "model.layers.{i}.mlp.experts.gate_proj.weight"`
- `experts_up_proj == "model.layers.{i}.mlp.experts.up_proj.weight"`
- `experts_down_proj == "model.layers.{i}.mlp.experts.down_proj.weight"`

**Validates: Requirements 3.5**

### Property 4: 零权重 DecoderLayer（层索引 ≥ 1）的输出等于输入 residual

*For any* 输入 tensor（任意形状和值），以零权重初始化的 `DecoderLayer`（层索引 ≥ 1）调用 `forward` 后，输出 tensor 的每个元素在 `f16::to_bits()` 比较下必须与输入 residual 完全相等。

**Validates: Requirements 4.6, 8.3**

### Property 5: 所有已知配置文件均可成功加载

*For any* 已知模型配置文件（Qwen3-Coder-30B、Llama-2-7b、Llama-2-70b、MiniMax-M2.5），`Config::load_from_file` 必须返回 `Ok(_)`，不得 panic 或返回 `Err`。

**Validates: Requirements 8.4**

### Property Reflection（冗余分析）

- Property 1（layers 长度）和 Property 2（FfnKind 优先级）是独立的，不互相蕴含。
- Property 3（名称格式）和 Property 4（零权重输出）测试不同层面，无冗余。
- Property 5（配置加载）是对 Property 1 和 2 的前提条件，但测试的是更广泛的加载成功性，保留。
- 五个属性各自提供独立的验证价值，无需合并或删除。

---

## Error Handling

### 配置加载错误

| 错误场景 | 处理方式 |
|---------|---------|
| JSON 文件不存在 | 返回 `Err`（来自 `File::open`） |
| JSON 格式错误 | 返回 `Err`（来自 `serde_json::from_reader`） |
| 缺少必要字段（如 `hidden_size`） | 返回 `Err`，错误消息包含字段名 |
| `layers.len() != num_hidden_layers` | 返回 `Err`，说明不一致 |

### SparseMoe 构造错误

| 错误场景 | 处理方式 |
|---------|---------|
| `use_routing_bias=true` 但 `router_bias=None` | `expect()` panic，附带说明消息 |

### AttentionBlock::Linear

`Linear` 变体作为占位符存在。若 `forward` 被调用，panic 并输出：
```
"Linear attention is not implemented in this phase"
```

### 不变量违反

`DecoderLayer::new` 中 `FfnKind` 与 `FfnTensorNames` 不匹配时，使用 `unreachable!()` 处理（已在现有代码中实现）。

---

## Testing Strategy

### 测试框架

使用 Rust 内置测试框架（`#[test]`）。属性测试使用 [`proptest`](https://github.com/proptest-rs/proptest) crate（已是 Rust 生态标准选择）。

### 单元测试（example-based）

每个模块的现有测试保持不变，并补充以下新测试：

**`config::tests`**：
- `test_from_file`：加载 Qwen3 配置，验证 `layers.len() == num_hidden_layers`（已存在）
- `test_load_all_known_configs`：加载全部 4 个已知配置文件，验证均返回 `Ok`
- `test_qwen3_layer_plan`：验证 Qwen3-Coder-30B 的每层 `FfnKind` 均为 `SparseMoe`，且 `router_scoring == Softmax`，`use_routing_bias == false`
- `test_missing_field_error`：传入缺少 `hidden_size` 的 JSON，验证 `Err` 包含字段名

**`names::tests`**（新增）：
- `test_qwen_layer_names_sparse_moe`：验证 Qwen 族 SparseMoe 层的名称格式
- `test_qwen_layer_names_dense`：验证 Dense 层的名称格式
- `test_model_tensor_names`：验证 `model_tensor_names` 返回正确字段

**`sparse_moe::tests`**（现有，保持不变）：
- `test_sparse_moe_queue_structure`
- `test_sparse_moe_sigmoid_queue_structure`
- `test_sparse_moe_zero_weights_output_equals_residual_bits`
- `test_sparse_moe_single_thread_equals_multi_thread_bits`

**`decoder_layer::test`**（现有，保持不变）：
- `test_decoder_layer_f32`
- `test_decoder_layer_f16`

**`model::test`**（现有，保持不变）：
- `test_model_forward`
- `test_model_forward_f16`

### 属性测试（property-based）

使用 `proptest` crate，每个属性测试运行最少 100 次迭代。

**Property 1 测试**（`config::tests`）：
```rust
// Feature: sparse-moe-refactor, Property 1: layers length equals num_hidden_layers
proptest! {
    #[test]
    fn prop_layers_len_equals_num_hidden_layers(
        num_layers in 1usize..=10usize,
        // 生成合法的最小 HfConfig
    ) {
        // 构造 HfConfig，验证 ResolvedConfig.layers.len() == num_layers
    }
}
```

**Property 2 测试**（`config::tests`）：
```rust
// Feature: sparse-moe-refactor, Property 2: FfnKind priority rule
proptest! {
    #[test]
    fn prop_ffn_kind_priority_rule(
        layer_idx in 0usize..=63usize,
        num_experts in 0usize..=256usize,
        mlp_only_layers in proptest::collection::vec(0usize..=63usize, 0..=10),
    ) {
        // 验证每层的 FfnKind 遵循优先级规则
    }
}
```

**Property 3 测试**（`names::tests`）：
```rust
// Feature: sparse-moe-refactor, Property 3: Qwen SparseMoe tensor name format
proptest! {
    #[test]
    fn prop_qwen_sparse_moe_tensor_names(layer_idx in 0usize..=63usize) {
        // 构造 Qwen SparseMoe 配置，验证名称格式
    }
}
```

**Property 4 测试**（`decoder_layer::test`）：
```rust
// Feature: sparse-moe-refactor, Property 4: zero-weight layer output equals residual
proptest! {
    #[test]
    fn prop_zero_weight_layer_output_equals_residual(
        batch_size in 1usize..=8usize,
        // hidden_states values
    ) {
        // 构造零权重 DecoderLayer（idx=1），验证输出 bits == 输入 bits
    }
}
```

**Property 5 测试**（`config::tests`）：
```rust
// Feature: sparse-moe-refactor, Property 5: all known configs load successfully
// 注：此属性的输入空间是有限的已知文件集合，使用参数化测试而非 proptest
#[test]
fn test_all_known_configs_load() {
    let paths = [
        "models/Qwen3-Coder-30B-A3B-Instruct/config.json",
        "models/Llama-2-7b-hf/config.json",
        "models/Llama-2-70b-hf/config.json",
        "models/MiniMax-M2.5/config.json",
    ];
    for path in &paths {
        assert!(Config::load_from_file(path).is_ok(), "Failed to load: {}", path);
    }
}
```

### 回归安全网

重构的每个步骤完成后，运行 `cargo test --workspace` 确保所有现有测试通过。关键回归测试：

1. `sparse_moe::tests` — 验证 SparseMoe 行为不变
2. `decoder_layer::test` — 验证 DecoderLayer 行为不变
3. `model::test::test_model_forward` — 端到端验证
4. `config::tests::test_from_file` — 验证配置加载不变
