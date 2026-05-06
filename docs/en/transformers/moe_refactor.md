# MoE Directory Refactor Proposal

## Background

The current `src/transformer` directory is responsible for several kinds of concerns at the same time:

1. The general decoder-only model skeleton: `model.rs`, `decoder_layer.rs`, `attention.rs`, `rope.rs`
2. Qwen3-MoE-specific FFN structure: `sparse_moe_block.rs`
3. Direct reads of HuggingFace config
4. Assumptions about concrete weight names: `q_proj/k_proj/v_proj/o_proj`, `mlp.experts.*`, `model.embed_tokens.weight`, `lm_head.weight`

As a result, the current implementation is not really "MoE infrastructure." It is "a Qwen3-MoE model implementation." So once you want to support the following model families, you immediately hit extension problems:

* Dense MLP models such as Llama / Mistral
* Sparse MoE models such as Mixtral / Qwen3-MoE
* Shared experts, hierarchical routing, and variants with different RoPE settings
* Different block topologies and weight-name conventions under the same operator backend

There are already two signals in the repository that show this is a real need rather than premature design:

* `models/` contains both Llama and Qwen3-MoE configs
* `src/bin/main.rs` currently binds directly to `moe::config::Config` and `moe::model::Model`

## Current Problems

### 1. Module Naming and Responsibility Do Not Match

`src/transformer` contains attention, rope, decoder layer, and the whole model, none of which are MoE-specific capabilities. The directory name will keep misleading future extensions and will force dense models into `moe`.

### 2. `DecoderLayer` Is Hard-Coded to One Block Combination

The current `decoder_layer.rs` fixes the layer structure as:

`LookupRms -> Attention -> RMS -> SparseMoeBlock`

That means:

* Dense MLP cannot be plugged in
* `mlp_only_layers` is not modeled as a layer plan
* There is nowhere to express pre-norm/post-norm, shared experts, or router variants for different families

### 3. `Config` Is a Single Large Structure Mixing Fields from Multiple Models

The current `config.rs` contains:

* Fields needed by Llama
* Fields needed by Qwen3-MoE
* Fields that do not exist for some models at all

The problem is not that there are too many fields. The problem is that there is no normalized internal config. Treating the HuggingFace JSON structure as the runtime structure makes every module full of `Option`, defaults, and model-specific special cases.

### 4. Model Construction Is Coupled to Weight Naming

`model.rs`, `attention.rs`, and `sparse_moe_block.rs` directly construct tensor keys with specific names. That is fine for a single model, but when you support new models, the first thing that usually changes is:

* Block weight names
* Whether bias exists
* Layer-prefix naming for MLP / MoE
* RoPE naming and shape conventions

If you do not separate "structure" from "name mapping," every new model will add more `if model_type == ...` logic into the core path.

## Refactoring Goals

The goal is not to build an over-abstracted framework. The goal is to satisfy four things:

1. The same runtime/operator backend can assemble different decoder block structures
2. Dense and sparse MoE FFNs can both be supported
3. HuggingFace raw config is first converted into an internal unified config before model construction
4. When adding a new model family, prefer adding an adapter layer rather than changing the core execution path

## Design Principles

### 1. Move the General Skeleton Up, Push Model Features Down

* General skeleton: embedding, rope, attention, decoder stack, lm head
* Model features: layer plan, FFN type, weight naming, config mapping

### 2. Avoid Overusing Trait Objects in Hot Paths

Runtime hot paths should still be implemented with static dispatch or lightweight enum dispatch. It is not recommended to make every forward pass into a deep `Box<dyn Trait>` object tree.

Recommended approach:

* Use traits / factories during construction
* Use enums to hold concrete blocks during execution

### 3. Split Config into Two Layers

* Outer layer: parse the raw HuggingFace config
* Inner layer: a unified internal model spec `ModelSpec`

### 4. Model Weight Naming Separately

Weight-name mapping is part of model-family adaptation and should not be scattered across attention / MLP / model code.

## Recommended Directory Structure

It is recommended to gradually migrate `src/transformer` to `src/model` while keeping compatibility exports for some time.

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

If you do not want to rename the public module right away, you can keep `src/transformer` for now and reorganize internally according to the layers above, then rename it in the second phase.

## Core Abstractions

### 1. Unified Model Spec `ModelSpec`

`ModelSpec` is the only model description the runtime depends on. It does not directly expose the raw HuggingFace JSON structure.

Recommended fields:

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

This way, dense and MoE differences are reflected in `LayerSpec`, rather than being hard-coded into `DecoderLayer`.

### 2. `DecoderBlock` Uses Enums to Combine Concrete FFNs

It is recommended to turn the current `DecoderLayer` into a generic `DecoderBlock`:

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

Benefits:

* Dense and MoE share the same decoder-block skeleton
* Rules such as `mlp_only_layers` can be decided when building `Vec<LayerSpec>`
* Shared expert / gated MLP support only needs `FfnBlock` extension

### 3. Add a `FamilyAdapter` for Each Model Family

It is recommended to add a layer of model-family adapters that converts "raw config + weight-naming rules" into a unified internal structure.

```rust
pub trait FamilyAdapter {
    fn matches(config: &HfConfig) -> bool;
    fn build_spec(config: &HfConfig) -> anyhow::Result<ModelSpec>;
    fn names(spec: &ModelSpec) -> TensorNames;
}
```

Responsibilities:

* `adapter.rs`: map a family's config to `ModelSpec`
* `names.rs`: provide the tensor naming rules for that family

After this, the difference between Llama and Qwen mostly stays under `families/` instead of polluting `core/`.

### 4. Separate Weight Naming from Constructors

The current `Attention::new`, `SparseMoeBlock::new`, and `Model::new` build tensor names themselves. It is better to pass in a name description object from the outside.

Example:

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

The component only cares about "which tensors it needs" and not "what they are called in some family."

## File-Level Migration Suggestions

### `config.rs`

Split into two layers:

* `spec/hf_config.rs`: parse HuggingFace config as faithfully as possible, with `Option` fields allowed
* `spec/model_spec.rs`: internal unified spec with complete fields, directly usable for model construction

Do not keep letting runtime modules read the raw HuggingFace fields directly.

### `model.rs`

Responsibilities become:

* Read `ModelSpec`
* Build embeddings, decoder blocks, final norm, and lm head
* Do not care about family-specific differences

Special cases currently in `Model::new` should move into adapters / factories.

### `decoder_layer.rs`

Rename it to `decoder_block.rs`, remove the direct dependency on `SparseMoeBlock`, and make it hold `FfnBlock`.

### `attention.rs`

Keep it as a generic attention component, but move these model differences out of the code:

* Whether qkv bias exists
* The source of num kv heads and grouped query attention config
* Possible qk norm / sliding window / causal mask variants

`AttentionSpec` should describe the differences, and `AttentionBlock` should execute them.

### `sparse_moe_block.rs`

Keep it as a component, but move it to `components/ffn/sparse_moe.rs`, and let it accept only:

* `SparseMoeSpec`
* `SparseMoeTensorNames`

instead of a list of scattered parameters from one concrete model config.

### `mlp.rs`

Restore it as a first-class citizen and move it to `components/ffn/dense_mlp.rs`. If dense models are not first-class, dense support will remain a special case forever.

### `rope.rs`

Keep it as a generic component, but let `RopeSpec` describe the parameters. If linear scaling, dynamic NTK, YaRN, or other variants are added later, this layer will have room for them.

## Build Flow Suggestions

It is recommended to fix model instantiation into the following 5 steps:

1. Read HuggingFace `config.json` into `HfConfig`
2. Choose a `FamilyAdapter` through `registry`
3. Let the adapter generate a unified `ModelSpec`
4. Let the adapter generate `TensorNames`
5. Build `CausalLm` with `ModelSpec + TensorNames`

Then, when adding a new model, the things you add first should be:

* One adapter
* One names set
* A small amount of necessary component implementation

instead of changing the public logic of `Model`, `DecoderLayer`, and `Attention`.

## Recommended Migration Phases

### Phase 1: Reorganize First, Do Not Change Behavior

Goal: do not affect the current Qwen3-MoE path.

Suggested steps:

1. Introduce `ModelSpec`, `LayerSpec`, and `FfnSpec`
2. Change `DecoderLayer` into `DecoderBlock + FfnBlock`
3. Move `SparseMoeBlock` and `MLP` into a unified `ffn/` directory
4. Add a Qwen adapter for `HfConfig -> ModelSpec`
5. Keep the external call surface of `src/bin/main.rs` as unchanged as possible

After this stage, the code structure already supports both dense and sparse paths, but only Qwen3-MoE is wired up.

### Phase 2: Add the Llama Dense Path

Suggested steps:

1. Implement `families/llama/adapter.rs`
2. Generate `FfnSpec::Dense` in the layer plan
3. Use `dense_mlp.rs` to build the corresponding block
4. Verify the common path for embeddings, rope, attention, and final norm

Only after this can we prove that the refactor really compressed model differences into the adapter layer rather than just renaming directories.

### Phase 3: Converge Module Naming

Suggested steps:

1. Rename `src/transformer` to `src/model`
2. Re-export it compatibly in `lib.rs` for a period of time
3. Update references in `bin/`, tests, and documentation

This step is best done after behavior is stable so that "architecture reorganization" and "public API rename" are not mixed into the same commit.

## Minimal Implementation Checklist

If you only do one minimal round of refactoring, I recommend prioritizing the following instead of rewriting everything at once:

1. Define `HfConfig` and `ModelSpec`
2. Define `FfnSpec` and `FfnBlock`
3. Change `DecoderLayer` into a generic block
4. Make `sparse_moe_block.rs` and `mlp.rs` parallel components
5. Move tensor-name concatenation into `names.rs`
6. Add `QwenAdapter` and move the current logic there
7. Then add `LlamaAdapter` to prove the abstraction works

## What Not to Do

### 1. Do Not Keep Adding Fields to `Config`

That only makes `config.rs` look more and more like a "dumping ground for all model JSON fields," while the runtime still lacks stable internal semantics.

### 2. Do Not Add More `if model_type == ...` Inside `DecoderLayer::new`

This is one of the easiest ways to make the code unmaintainable. Layer-structure selection must be moved earlier into the spec/build stage.

### 3. Do Not Put All Differences into Trait Objects

Model-family adaptation can use traits, but the actual execution path should keep a clear data structure and predictable dispatch whenever possible.

## Expected Benefits

After the refactor, the code will gain three direct benefits:

1. Lower cost of adding support for new models; new models mainly require adapters and names
2. Clearer general execution skeleton; the boundary between `core` and `families` becomes explicit
3. Even if we later support Mixtral, DeepSeek-MoE, shared experts, or different RoPE variants, we will not need to rewrite the backbone path

## Conclusion

The key to this refactor is not splitting the `moe` directory into more pieces. It is separating three things:

* The general decoder execution skeleton
* Model-family config and weight-name adaptation
* FFN type differences (`dense` / `sparse MoE` / `shared experts`)

If we only reorganize directories without introducing the three core abstractions `ModelSpec + FamilyAdapter + FfnBlock`, we will eventually fall back to piling model-specific exceptions into the public path again.
