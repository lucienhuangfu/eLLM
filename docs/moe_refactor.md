# moe 目录重构方案

## 背景

当前 `src/transformer` 目录里同时承担了几类职责：

1. 通用的 decoder-only 模型骨架：`model.rs`、`decoder_layer.rs`、`attention.rs`、`rope.rs`
2. Qwen3-MoE 特定的 FFN 结构：`sparse_moe_block.rs`
3. HuggingFace 配置直读：`config.rs`
4. 对具体权重命名的假设：`q_proj/k_proj/v_proj/o_proj`、`mlp.experts.*`、`model.embed_tokens.weight`、`lm_head.weight`

这导致当前实现本质上不是 “MoE 基础设施”，而是 “Qwen3-MoE 的一套模型实现”，所以一旦要支持下面这些模型族，就会立刻遇到扩展问题：

- Llama / Mistral 这类 dense MLP 模型
- Mixtral / Qwen3-MoE 这类 sparse MoE 模型
- 共享专家、分层路由、不同 RoPE 配置的变体
- 同一算子后端下，不同 block 拓扑和不同权重命名方式

仓库里已经有两个信号说明这个问题是现实需求，而不是提前设计：

- `models/` 下同时存在 Llama 和 Qwen3-MoE 配置
- `src/bin/main.rs` 当前直接绑定 `moe::config::Config` 和 `moe::model::Model`

## 当前问题

### 1. 模块命名和职责不一致

`src/transformer` 里包含 attention、rope、decoder layer、整个 model，这些都不是 MoE 专属能力。目录名会持续误导后续扩展，导致 dense 模型也不得不塞进 `moe`。

### 2. `DecoderLayer` 被写死为一种 block 组合

当前 `decoder_layer.rs` 的层结构固定为：

`LookupRms -> Attention -> RMS -> SparseMoeBlock`

这意味着：

- dense MLP 无法接入
- `mlp_only_layers` 没有被建模为层级计划
- 不同模型族的 pre-norm/post-norm、shared experts、router 变体都无处表达

### 3. `Config` 是单一大结构，混合了多模型字段

当前 `config.rs` 同时包含：

- Llama 需要的字段
- Qwen3-MoE 需要的字段
- 部分字段对某些模型根本不存在

问题不是字段多，而是缺少 “规范化后的内部配置”。直接把 HuggingFace JSON 结构当成运行时结构，会让每个模块都充满 `Option`、默认值和模型特例。

### 4. 模型构建逻辑和权重命名耦合

`model.rs`、`attention.rs`、`sparse_moe_block.rs` 中直接构造了特定名字的 tensor key。这对单一模型没问题，但支持新模型时，往往最先变化的就是：

- block 内权重名字
- 是否存在 bias
- MLP / MoE 的层前缀命名
- rotary embedding 的命名和形状约定

如果不把 “结构” 和 “命名映射” 拆开，每支持一个模型就会把更多 `if model_type == ...` 写进核心路径。

## 重构目标

重构后的目标不是做一个过度抽象的框架，而是满足下面四点：

1. 同一套 runtime/operator 后端，能装配不同 decoder block 结构
2. 能同时支持 dense 与 sparse MoE 两类 FFN
3. HuggingFace 原始配置先转换为内部统一配置，再进入模型构建
4. 新增模型族时，优先新增适配层，而不是修改核心执行路径

## 设计原则

### 1. 通用骨架上移，模型特性下沉

- 通用骨架：embedding、rope、attention、decoder stack、lm head
- 模型特性：layer 计划、ffn 类型、权重命名、配置映射

### 2. 热路径避免 trait object 滥用

运行时热点仍然应以静态分发或轻量 enum 分发为主，不建议把每层前向都做成 `Box<dyn Trait>` 的深层对象树。

推荐方式：

- 构建阶段使用 trait / factory
- 运行阶段使用 enum 持有具体 block

### 3. 配置分两层

- 外层：原始 HuggingFace 配置解析
- 内层：统一的内部模型规格 `ModelSpec`

### 4. 权重命名单独建模

权重名映射是模型族适配的一部分，不应该散落在 attention / mlp / model 各处。

## 建议目录结构

建议把 `src/transformer` 逐步迁移为 `src/model`，并保留一段时间兼容导出。

```text
src/
  model/
    mod.rs
    registry.rs
    spec/
      mod.rs
      hf_config.rs
      model_spec.rs
      layer_spec.rs
    core/
      mod.rs
      causal_lm.rs
      decoder_block.rs
      embeddings.rs
    components/
      mod.rs
      attention.rs
      rope.rs
      norm.rs
      ffn/
        mod.rs
        dense_mlp.rs
        sparse_moe.rs
    families/
      mod.rs
      llama/
        mod.rs
        adapter.rs
        names.rs
      qwen/
        mod.rs
        adapter.rs
        names.rs
      mixtral/
        mod.rs
        adapter.rs
        names.rs
```

如果短期内不想改公开模块名，也可以先保留 `src/transformer`，但按上面的层次在内部重组，再在第二阶段改名。

## 核心抽象

### 1. 统一模型规格 `ModelSpec`

`ModelSpec` 是运行时唯一依赖的模型描述，不直接暴露 HuggingFace 的原始 JSON 结构。

建议包含：

```rust
pub struct ModelSpec {
    pub family: ModelFamily,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub attention: AttentionSpec,
    pub rope: RopeSpec,
    pub final_norm: NormSpec,
    pub lm_head: LmHeadSpec,
    pub layers: Vec<LayerSpec>,
}

pub struct LayerSpec {
    pub attn: AttentionLayerSpec,
    pub ffn: FfnSpec,
    pub input_norm: NormSpec,
    pub post_attn_norm: NormSpec,
}

pub enum FfnSpec {
    Dense(DenseMlpSpec),
    SparseMoe(SparseMoeSpec),
}
```

这样 dense 与 MoE 的差异会体现在 `LayerSpec`，而不是体现在 `DecoderLayer` 的硬编码分支里。

### 2. `DecoderBlock` 使用 enum 组合具体 FFN

建议把当前 `DecoderLayer` 改造成通用 `DecoderBlock`：

```rust
pub struct DecoderBlock<T> {
    input_norm: NormLayer<T>,
    attention: AttentionBlock<T>,
    post_attn_norm: NormLayer<T>,
    ffn: FfnBlock<T>,
}

pub enum FfnBlock<T> {
    Dense(DenseMlp<T>),
    SparseMoe(SparseMoeBlock<T>),
}
```

优点：

- dense 和 MoE 共用一套 decoder block 骨架
- `mlp_only_layers` 这种规则在构建 `Vec<LayerSpec>` 时就能决定
- 新增 shared expert / gated MLP 时只需扩展 `FfnBlock`

### 3. 为模型族增加 `FamilyAdapter`

建议增加一层模型族适配器，把 “原始 config + 权重命名规则” 变成统一的内部结构。

```rust
pub trait FamilyAdapter {
    fn matches(config: &HfConfig) -> bool;
    fn build_spec(config: &HfConfig) -> anyhow::Result<ModelSpec>;
    fn names(spec: &ModelSpec) -> TensorNames;
}
```

职责划分：

- `adapter.rs`：把某个模型族的配置映射到 `ModelSpec`
- `names.rs`：提供该模型族的 tensor 命名规则

这样做以后，Llama 和 Qwen 的区别主要留在 `families/` 下，而不是污染 `core/`。

### 4. 把权重命名从构造函数中抽离

当前 `Attention::new`、`SparseMoeBlock::new`、`Model::new` 自己拼接 tensor 名字。建议改为外部传入一个名字描述对象。

示意：

```rust
pub struct LayerTensorNames {
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub input_norm: String,
    pub post_attn_norm: String,
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

组件只关心 “我需要哪些 tensor”，不关心 “它们在某个模型里叫什么”。

## 文件级迁移建议

### `config.rs`

拆成两层：

- `spec/hf_config.rs`：尽量忠实解析 HuggingFace 配置，字段允许 `Option`
- `spec/model_spec.rs`：内部统一规格，字段完整、可直接用于构建模型

不要继续让运行时各模块直接读取 HuggingFace 原始字段。

### `model.rs`

职责改为：

- 读取 `ModelSpec`
- 构建 embeddings、decoder blocks、final norm、lm head
- 不关心模型族差异

当前 `Model::new` 里的模型特例应当迁到 adapter/factory。

### `decoder_layer.rs`

重命名为 `decoder_block.rs`，去掉对 `SparseMoeBlock` 的直接依赖，改为持有 `FfnBlock`。

### `attention.rs`

保留为通用 attention 组件，但需要把下面这些模型差异从代码里挪出去：

- qkv bias 是否存在
- num kv heads 与 grouped query attention 的配置来源
- 可能的 qk norm / sliding window / causal mask 变体

`AttentionSpec` 负责描述差异，`AttentionBlock` 负责执行。

### `sparse_moe_block.rs`

保留为组件，但位置调整到 `components/ffn/sparse_moe.rs`，并让它只接受：

- `SparseMoeSpec`
- `SparseMoeTensorNames`

而不是接受一串来自某个具体模型配置的离散参数。

### `mlp.rs`

恢复为一等公民，迁到 `components/ffn/dense_mlp.rs`。当前如果只保留 sparse MoE，会让 dense 模型支持永远成为特例。

### `rope.rs`

作为通用组件保留，但 RoPE 参数应由 `RopeSpec` 描述。后续如果支持线性缩放、动态 NTK、YaRN 等变体，这一层就有扩展位。

## 构建流程建议

建议把模型实例化流程固定成下面 5 步：

1. 读取 HuggingFace `config.json` 为 `HfConfig`
2. 通过 `registry` 选择 `FamilyAdapter`
3. 由 adapter 生成统一的 `ModelSpec`
4. 由 adapter 生成 `TensorNames`
5. 用 `ModelSpec + TensorNames` 构建 `CausalLm`

这样新增模型时，优先新增的是：

- 一个 adapter
- 一套 names
- 少量必要组件实现

而不是修改 `Model`、`DecoderLayer`、`Attention` 的公共逻辑。

## 推荐迁移阶段

### 第一阶段：先重组，不改行为

目标：不影响现有 Qwen3-MoE 跑通路径。

建议操作：

1. 引入 `ModelSpec`、`LayerSpec`、`FfnSpec`
2. 把 `DecoderLayer` 改成 `DecoderBlock + FfnBlock`
3. 把 `SparseMoeBlock`、`MLP` 迁到统一 `ffn/` 目录
4. 新增 `HfConfig -> ModelSpec` 的 Qwen adapter
5. 保持 `src/bin/main.rs` 的外部调用面尽量不变

这一阶段完成后，代码结构已经支持 dense / sparse 两条分支，但先只接通 Qwen3-MoE。

### 第二阶段：接入 Llama dense 路径

建议操作：

1. 实现 `families/llama/adapter.rs`
2. 让 layer 规划生成 `FfnSpec::Dense`
3. 使用 `dense_mlp.rs` 构建对应 block
4. 验证 embeddings、rope、attention 与 final norm 的公共路径

完成后，才能证明这次重构真正把模型差异压缩到了 adapter 层，而不是只是重命名目录。

### 第三阶段：收敛模块命名

建议操作：

1. 把 `src/transformer` 改名为 `src/model`
2. 在 `lib.rs` 中以兼容方式 re-export 一段时间
3. 更新 `bin/`、测试、文档中的引用

这一步最好在行为稳定后做，避免把“架构重组”和“对外 API 改名”混在同一个提交里。

## 最小可落地实施清单

如果只做最小一轮改造，我建议优先做下面这些，而不是一次性全量重写：

1. 定义 `HfConfig` 和 `ModelSpec`
2. 定义 `FfnSpec` 与 `FfnBlock`
3. 把 `DecoderLayer` 改成通用 block
4. 把 `sparse_moe_block.rs` 与 `mlp.rs` 并列化
5. 把 tensor 名称拼接移到 `names.rs`
6. 新增 `QwenAdapter`，把现有逻辑迁进去
7. 再新增 `LlamaAdapter` 证明抽象有效

## 不建议的做法

### 1. 不要继续在 `Config` 上叠字段

这只会让 `config.rs` 越来越像 “所有模型 JSON 字段的大杂烩”，但运行时仍然拿不到稳定的内部语义。

### 2. 不要在 `DecoderLayer::new` 里写更多 `if model_type == ...`

这是最容易走向不可维护的地方。层结构选择必须前置到 spec/build 阶段。

### 3. 不要把所有差异都塞进 trait object

模型族适配可以用 trait，但真正执行路径应尽量保持明确的数据结构和可预测分发。

## 预期收益

完成后，这套代码会得到三个直接收益：

1. 支持多模型的新增成本下降，新增模型主要是写 adapter 和 names
2. 通用执行骨架更清晰，`core` 与 `families` 的职责边界明确
3. 后续即使要支持 Mixtral、DeepSeek-MoE、共享专家或不同 RoPE 变体，也不需要重写主干路径

## 结论

这次重构的关键不是把 `moe` 目录拆得更细，而是把三个东西分开：

- 通用 decoder 执行骨架
- 模型族配置与权重命名适配
- FFN 类型差异（dense / sparse MoE / 共享专家）

如果只做目录整理但不引入 `ModelSpec + FamilyAdapter + FfnBlock` 这三个核心抽象，最终还是会回到在公共路径里堆模型特例的状态。