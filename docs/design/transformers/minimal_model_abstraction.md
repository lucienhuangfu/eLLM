# Minimal Model Abstraction

## Goal

The goal of this proposal is not to refactor the current `src/transformer` directory into a complete framework in one step. Instead, it aims to turn the current implementation from a "Qwen3-MoE-only path" into a "general decoder path that can be extended to dense / sparse MoE / hybrid attention" while changing as little existing code as possible.

The proposal must satisfy all of the following:

1. Keep the current Qwen3-MoE execution path basically unchanged
2. Be able to support dense MLP models such as Llama / Mistral later
3. Be able to support MoE models such as Mixtral / MiniMax-M2.5 later
4. Avoid introducing overly heavy abstraction in the first stage

## Design Basis

This proposal is based on two constraints:

### 1. Experience from Hugging Face Transformers

Hugging Face does not first normalize every model into a single IR and then assemble it with one universal builder. Instead, it:

1. Dispatches by `model_type` / family
2. Gives each family its own config
3. Reuses similar models through inheritance and local overrides
4. Expresses layer differences ahead of time via fields in the config, such as `layer_types`

So the core lesson is not a "grand unified spec," but:

1. family-first
2. layer-plan-first
3. unified entry point, localized differences

### 2. The Reality of This Repository

The current `src/transformer` already has:

1. `attention.rs`
2. `decoder_layer.rs`
3. `sparse_moe_block.rs`
4. `mlp.rs`
5. `model.rs`

The real problem is not that the directory is too shallow. The problem is that there are three tight couplings:

1. `config.rs` directly carries the JSON structure of one concrete model
2. `decoder_layer.rs` is hard-coded as `Attention + SparseMoeBlock`
3. Tensor-name concatenation is scattered across model / attention / block constructors

So the simplest and most effective first-stage solution is not to split the directory first, but to split these three couplings first.

## Minimal Abstraction

The minimal solution introduces only 3 new abstractions:

1. `ModelFamily`
2. `LayerPlan`
3. `TensorNames`

These three abstractions are enough to cover:

1. dense MLP
2. sparse MoE
3. hybrid attention
4. naming differences across families

## Abstraction 1: `ModelFamily`

`ModelFamily` is used for family identification and dispatch upfront.

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

The purpose of this layer is:

1. Not to fill the runtime core logic with `if model_type == ...`
2. To concentrate family differences in the config resolution and name resolution stages

At this stage there is no need for traits; a plain `match` is enough:

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

## Abstraction 2: `LayerPlan`

`LayerPlan` is the core of this minimal proposal.

The current problem with `DecoderLayer` is not its implementation details, but the fact that it hard-codes the layer structure. To support models like MiniMax / MiniMax-M2.5, the "how each layer runs" decision must be lifted into data.

Recommended definition:

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

This is enough to express:

1. Llama: `Full + Dense`
2. Qwen3-MoE: `Full + SparseMoe`
3. Mixtral: `Full + SparseMoe`
4. MiniMax: mixed `Full/Linear` + `SparseMoe`
5. MiniMax-M2.5: as long as it fits these attention/ffn combinations, it can be plugged in

## Abstraction 3: `TensorNames`

The current implementation hard-codes tensor keys directly in constructors, which is the weakest point when supporting multiple models.

The minimal solution only needs a lightweight naming description object:

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

Then the components only care about "which tensors do I need" and not "what are they called in this family."

## Split Config into Two Layers, Not a Full `ModelSpec`

To keep the cost under control, I do not recommend introducing a full `ModelSpec` immediately. Two layers are enough for now:

1. `HfConfig`: maps Hugging Face's original JSON as faithfully as possible, allowing `Option`
2. `ResolvedConfig`: the stable runtime config actually used by the executor, with complete fields and no family-specific checks

Recommended definition:

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

The core principles are:

1. The raw JSON can be messy
2. The runtime config must be stable
3. Family-specific quirks are handled only in `HfConfig -> ResolvedConfig`

## Minimal Change to `DecoderLayer`

The current `DecoderLayer` directly owns:

1. `Attention`
2. `SparseMoeBlock`

The minimal change is not to introduce trait objects, but to switch to enums:

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

In the first stage, we do not even need to implement `LinearAttention`; we only need to leave room for it.

The benefits are:

1. Runtime remains a static structure plus lightweight enum dispatch
2. No deep trait object tree is needed
3. MiniMax support later will not require another layer-framework rewrite

## Why This Can Fit MiniMax M2.5

The main challenges MiniMax / MiniMax-M2.5 pose to the current code fall into two categories:

1. It may be a Mixtral-style sparse MoE
2. It may use hybrid attention, meaning different layers may use different attention types

Hugging Face handles MiniMax essentially by:

1. Describing the model with family config
2. Expressing layer differences ahead of time via `layer_types`
3. Reusing the existing Mixtral-style implementation for similar families

So as long as we have:

1. `ModelFamily`
2. `LayerPlan`
3. `TensorNames`

MiniMax-M2.5 is just one more resolver:

```rust
fn resolve_minimax_m25(hf: &HfConfig) -> anyhow::Result<ResolvedConfig>
```

It is responsible for:

1. Identifying the family
2. Generating `LayerPlan` for each layer
3. Generating the corresponding `TensorNames`

The core execution skeleton does not need to change again.

## Recommended Directory Shape

At this stage, I do not recommend immediately renaming `src/transformer` into a full `src/model/core/components/families/spec` tree.

The minimal usable directory shape is:

```text
src/transformer/
  mod.rs
  config.rs
  names.rs
  attention.rs
  decoder_layer.rs
  mlp.rs
  sparse_moe_block.rs
  model.rs
```

Where:

1. `config.rs`: `HfConfig + ResolvedConfig + resolve_xxx`
2. `names.rs`: generates tensor names per family
3. `decoder_layer.rs`: holds attention/ffn enums according to `LayerPlan`

## Minimum First-Stage Implementation Order

### 1. Split `Config` into Two Layers

Split the current single `Config` into:

1. `HfConfig`
2. `ResolvedConfig`

At the same time, keep the old calling interface for compatibility, for example:

```rust
pub type Config = ResolvedConfig;
```

This minimizes changes to `main.rs` and the current call sites.

### 2. Restore `mlp.rs` as a Formal Module

`mlp.rs` already exists, but it is not yet part of the public path. In the first stage, dense FFN must become a formal first-class citizen.

### 3. Introduce `layers: Vec<LayerPlan>` in `ResolvedConfig`

This moves "how each layer runs" upfront instead of inferring it in `DecoderLayer::new`.

### 4. Change `DecoderLayer` to `AttentionBlock + FfnBlock`

But only implement:

1. `AttentionBlock::Full`
2. `FfnBlock::Dense`
3. `FfnBlock::SparseMoe`

`Linear` can stay as a placeholder for now.

### 5. Introduce `names.rs`

Pull tensor-name concatenation out of:

1. `model.rs`
2. `decoder_layer.rs`
3. `sparse_moe_block.rs`

### 6. In the First Stage, Only Wire Up Qwen3-MoE

The goal is not to support every model at once, but to prove:

1. Current behavior remains unchanged
2. The abstractions already allow dense and sparse paths to coexist

### 7. In the Second Stage, Add Llama Dense

This is the minimum requirement to validate that the abstraction works.

### 8. In the Third Stage, Add MiniMax / MiniMax-M2.5

At that point, you should only need new family resolution and name resolution, not another change to the shared layer skeleton.

## What Not to Do in the Current Stage

To keep complexity minimal, the following are not recommended in the first round:

1. Do not rename `src/transformer` to `src/model` immediately
2. Do not introduce a full `ModelSpec + FamilyAdapter + Registry` system right away
3. Do not turn all differences into trait objects immediately
4. Do not introduce multimodal abstractions immediately
5. Do not overdesign spec structures just to cover every future model

These can wait until the first restructuring round is stable.

## Final Recommendation

Considering implementation cost and extension benefit together, the most suitable landing approach is:

1. family-first
2. layer-plan-first
3. names-outside-components

That means:

1. Use `ModelFamily` to decide which parsing path to take
2. Use `LayerPlan` to decide each layer's block combination
3. Use `TensorNames` to absorb naming differences

These three layers are already enough to upgrade the current implementation from "Qwen3-MoE-only" to "a minimal general-purpose skeleton that can support Llama / Mixtral / MiniMax-M2.5", without rewriting the entire framework from scratch at the beginning.
