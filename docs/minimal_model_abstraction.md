# 最小模型抽象方案

## 目标

这份方案的目标不是一次性把当前 `src/moe` 重构成完整框架，而是在**尽量少改现有代码**的前提下，把当前实现从“Qwen3-MoE 专用路径”提升为“可扩展到 dense / sparse MoE / hybrid attention 的通用 decoder 路径”。

方案需要同时满足：

1. 保持当前 Qwen3-MoE 跑通路径基本不变
2. 后续能接入 Llama / Mistral 这类 dense MLP 模型
3. 后续能接入 Mixtral / MiniMax-M2.5 这类 MoE 模型
4. 不在第一阶段引入过重的抽象层

## 设计依据

这个方案结合了两方面约束：

### 1. Hugging Face Transformers 的经验

Hugging Face 的做法并不是先把所有模型规整成一个统一 IR，再由一个通用 builder 去装配，而是：

1. 按 `model_type` / family 做分发
2. 每个 family 有自己的 config
3. 相近模型通过继承和局部 override 复用
4. 层级差异通过 config 中的字段前置表达，例如 `layer_types`

这意味着值得借鉴的核心不是“大一统 spec”，而是：

1. family-first
2. layer-plan-first
3. 统一入口，局部差异下沉

### 2. 当前仓库的实际情况

当前 `src/moe` 已经有：

1. `attention.rs`
2. `decoder_layer.rs`
3. `sparse_moe_block.rs`
4. `mlp.rs`
5. `model.rs`

真正的问题不在于目录不够细，而在于三处耦合：

1. `config.rs` 直接承载某个具体模型的 JSON 结构
2. `decoder_layer.rs` 写死为 `Attention + SparseMoeBlock`
3. tensor 名称拼接分散在 model / attention / block 构造函数中

因此第一阶段最简单、最有效的做法，不是先拆目录，而是先拆这三个耦合点。

## 最小抽象

最小方案只引入 3 个新抽象：

1. `ModelFamily`
2. `LayerPlan`
3. `TensorNames`

这三个抽象就足以覆盖：

1. dense MLP
2. sparse MoE
3. hybrid attention
4. 不同 family 的权重命名差异

## 抽象 1：ModelFamily

`ModelFamily` 用来前置做 family 识别和分发。

```rust
pub enum ModelFamily {
    Llama,
    Qwen,
    Mixtral,
    MiniMax,
    MiniMaxM2,
    Unknown,
}
```

这层的作用是：

1. 不让运行时核心逻辑充满 `if model_type == ...`
2. 让 family 差异集中在 config resolve 和 names resolve 阶段

当前阶段不需要上 trait，直接 `match` 即可：

```rust
match detect_family(hf_config) {
    ModelFamily::Qwen => resolve_qwen(hf_config),
    ModelFamily::Llama => resolve_llama(hf_config),
    ModelFamily::Mixtral => resolve_mixtral(hf_config),
    ModelFamily::MiniMax => resolve_minimax(hf_config),
    ModelFamily::MiniMaxM2 => resolve_minimax_m2(hf_config),
    ModelFamily::Unknown => bail!("unsupported model family"),
}
```

## 抽象 2：LayerPlan

`LayerPlan` 是这个最小方案的核心。

当前 `DecoderLayer` 的问题不是实现细节，而是它把层结构写死了。为了支持 MiniMax / MiniMax-M2.5 这类模型，必须把“每层怎么跑”前置成数据。

建议定义：

```rust
pub struct LayerPlan {
    pub attention: AttentionKind,
    pub ffn: FfnKind,
}

pub enum AttentionKind {
    Full,
    SlidingWindow,
    Linear,
}

pub enum FfnKind {
    Dense {
        intermediate_size: usize,
    },
    SparseMoe {
        moe_intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
    },
}
```

这层足以表达：

1. Llama: `Full + Dense`
2. Qwen3-MoE: `Full + SparseMoe`
3. Mixtral: `Full + SparseMoe`
4. MiniMax: `Full/Linear` 混排 + `SparseMoe`
5. MiniMax-M2.5: 只要能落入这些 attention/ffn 组合，就能接入

## 抽象 3：TensorNames

当前实现把 tensor key 直接写死在构造函数里，这是支持多模型时最脆弱的点。

最小方案里只需要做一个轻量命名描述对象：

```rust
pub struct ModelTensorNames {
    pub embed_tokens: String,
    pub lm_head: String,
    pub layers: Vec<LayerTensorNames>,
}

pub struct LayerTensorNames {
    pub input_norm: String,
    pub post_attn_norm: String,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub ffn: FfnTensorNames,
}

pub enum FfnTensorNames {
    Dense {
        gate_proj: String,
        up_proj: String,
        down_proj: String,
    },
    SparseMoe {
        router: String,
        experts_gate: String,
        experts_up: String,
        experts_down: String,
    },
}
```

这样组件只关心“我需要哪些 tensor”，而不关心“这个 family 里它们叫什么”。

## Config 只拆两层，不上完整 ModelSpec

为了控制成本，当前阶段不建议直接引入完整 `ModelSpec`。先拆成两层就够：

1. `HfConfig`：尽量忠实映射 Hugging Face 原始 JSON，允许 `Option`
2. `ResolvedConfig`：运行时真正使用的稳定配置，字段完整，不再做 family 特例判断

建议定义：

```rust
pub struct HfConfig {
    pub model_type: String,
    pub architectures: Vec<String>,

    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: Option<usize>,
    pub moe_intermediate_size: Option<usize>,

    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,

    pub rms_norm_eps: f32,
    pub rope_theta: Option<f32>,
    pub max_position_embeddings: usize,

    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub norm_topk_prob: Option<bool>,
    pub qkv_bias: Option<bool>,

    pub mlp_only_layers: Option<Vec<usize>>,
    pub layer_types: Option<Vec<String>>,
    pub sliding_window: Option<usize>,
}

pub struct ResolvedConfig {
    pub family: ModelFamily,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub layers: Vec<LayerPlan>,
}
```

核心原则是：

1. 原始 JSON 可以脏
2. 运行时配置必须稳定
3. family 特例只在 `HfConfig -> ResolvedConfig` 这一步消化

## DecoderLayer 的最小改法

当前 `DecoderLayer` 直接持有：

1. `Attention`
2. `SparseMoeBlock`

最小改法不是上 trait object，而是改成 enum：

```rust
pub enum AttentionBlock<T> {
    Full(Attention<T>),
    LinearPlaceholder,
}

pub enum FfnBlock<T> {
    Dense(MLP<T>),
    SparseMoe(SparseMoeBlock<T>),
}

pub struct DecoderLayer<T> {
    attention: AttentionBlock<T>,
    ffn: FfnBlock<T>,
    ...
}
```

第一阶段甚至不需要实现 `LinearAttention`，只要把位置留出来即可。

这样做的好处是：

1. 运行时仍然是静态结构 + 轻量 enum 分发
2. 不需要深层 trait object
3. 后续支持 MiniMax 时不用再改 layer 框架

## 为什么这个方案可以适配 MiniMax M2.5

MiniMax / MiniMax-M2.5 类模型对当前代码的挑战主要有两类：

1. 它可能是 Mixtral 风格的 sparse MoE
2. 它可能有 hybrid attention，即不同层 attention 类型不同

Hugging Face 对 MiniMax 的处理本质上也是：

1. 用 family config 描述模型
2. 用 `layer_types` 前置表达层差异
3. 在相近 family 上复用已有 Mixtral 风格实现

因此，只要我们有：

1. `ModelFamily`
2. `LayerPlan`
3. `TensorNames`

MiniMax-M2.5 就只是多一个 resolver：

```rust
fn resolve_minimax_m25(hf: &HfConfig) -> anyhow::Result<ResolvedConfig>
```

它负责：

1. 识别 family
2. 生成每层 `LayerPlan`
3. 生成对应 `TensorNames`

核心执行骨架不需要再改。

## 建议目录形态

当前阶段不建议直接把 `src/moe` 改成完整的 `src/model/core/components/families/spec` 目录树。

最小够用目录：

```text
src/moe/
  mod.rs
  config.rs
  names.rs
  attention.rs
  decoder_layer.rs
  mlp.rs
  sparse_moe_block.rs
  model.rs
```

其中：

1. `config.rs`：`HfConfig + ResolvedConfig + resolve_xxx`
2. `names.rs`：按 family 生成 tensor names
3. `decoder_layer.rs`：按 `LayerPlan` 持有 attention/ffn enum

## 第一阶段最小实施顺序

### 1. `Config` 改成两层

把当前单一 `Config` 拆成：

1. `HfConfig`
2. `ResolvedConfig`

同时保留原来的调用接口兼容，例如：

```rust
pub type Config = ResolvedConfig;
```

这样可以尽量少改 `main.rs` 和当前调用代码。

### 2. 恢复 `mlp.rs` 为正式模块

当前 `mlp.rs` 已经存在，但还不是公开路径。第一阶段必须让 dense FFN 成为正式一等公民。

### 3. 在 `ResolvedConfig` 中引入 `layers: Vec<LayerPlan>`

这一层把“每层怎么跑”前置，不再在 `DecoderLayer::new` 里推断。

### 4. `DecoderLayer` 改为 `AttentionBlock + FfnBlock`

但只实现：

1. `AttentionBlock::Full`
2. `FfnBlock::Dense`
3. `FfnBlock::SparseMoe`

`Linear` 可以先留占位。

### 5. 引入 `names.rs`

把 tensor 名称拼接从：

1. `model.rs`
2. `decoder_layer.rs`
3. `sparse_moe_block.rs`

中抽出来。

### 6. 第一阶段只先接通 Qwen3-MoE

目标不是一口气支持所有模型，而是先证明：

1. 当前行为不变
2. 抽象已经允许 dense / sparse 两条路并存

### 7. 第二阶段接入 Llama dense

这是验证抽象有效性的最低要求。

### 8. 第三阶段接入 MiniMax / MiniMax-M2.5

这时只需要新增 family resolve 和 names resolve，不应该再改公共 layer 骨架。

## 当前阶段不建议做的事

为了保持最小复杂度，下面这些不建议在第一轮就做：

1. 不要立即把 `src/moe` 改名成 `src/model`
2. 不要立即引入完整 `ModelSpec + FamilyAdapter + Registry` 体系
3. 不要立即把所有差异都做成 trait object
4. 不要立即做多模态抽象
5. 不要为了兼容未来所有模型，先设计过多 spec 结构

这些都可以等第一轮重组稳定后再做。

## 最终建议

如果从实现成本和扩展收益综合看，当前最合适的落地方案是：

1. family-first
2. layer-plan-first
3. names-outside-components

也就是：

1. 用 `ModelFamily` 决定走哪条解析路径
2. 用 `LayerPlan` 决定每层 block 组合
3. 用 `TensorNames` 吸收权重命名差异

这三层已经足够让当前实现从“Qwen3-MoE 专用”升级到“能承载 Llama / Mixtral / MiniMax-M2.5 的最小通用骨架”，而不需要一开始就进入完整框架化重写。