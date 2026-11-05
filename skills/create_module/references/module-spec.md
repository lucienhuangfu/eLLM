# eLLM Transformer Module Reference

Lookup material for the `create_module` skill. The workflow lives in [SKILL.md](../SKILL.md); this file covers catalogs, wiring, and on-request extension patterns.

## Module Catalog (`src/transformer/`)

| File | Module | Role |
|------|--------|------|
| `attention/attention_enum.rs` | `AttentionBlock<T>` | Per-layer dispatch by raw string: `config.layer_types[i].contains("linear")` → `GatedDelta`, anything else → `Full` |
| `attention/full_attention.rs` | `FullAttention<T>` | GQA self-attention: `matmul3` (Q/K/V + qk-norm + RoPE, optionally the `attn_output_gate` half) → `attention` → `matmul_add` (o_proj + residual); owns the `NO_QK_NORM_MODELS` table |
| `attention/linear_attention.rs` | `GatedDeltaAttention<T>` | Gated DeltaNet: `in_proj_*` → `causal_conv_silu` → `recurrent_gated_delta_rule` → `out_proj` |
| `mlp/dense_mlp.rs` | `DenseMlp<T>` | SwiGLU MLP: `matmul`×2 (gate/up) → `silu_mul` → `matmul_add` (down + residual); the simplest complete example |
| `mlp/ffn_block.rs` | `FfnBlock<T>` | Inline sparse predicate (`mlp_only_layers` / `num_experts` / `decoder_sparse_step`) → `Dense` or `SparseMoe` |
| `mlp/sparse_moe/` | `SparseMoe<T>` | MoE block in its own directory: `layer.rs` (public, dispatches the router on `scoring_func.contains("sigmoid")`), `router_softmax.rs` / `router_sigmoid.rs` (`pub(super)` variants behind a private `SparseMoeRouter` enum), `tests.rs` |
| `decoder_layer.rs` | `DecoderLayer<T>` | Assembles one layer: input layernorm (`rms` / `lookup_rms` at layer 0) → `AttentionBlock` → post-attention layernorm → `FfnBlock`; owns the `ZERO_CENTERED_MODELS` table |
| `rope.rs` | `RotaryEmbedding` | Pure CPU precompute of the position-embedding table (no tensors/operators; `forward<T>() -> Vec<T>`) |
| `text_model.rs` | `TextModel<T>` | Family-agnostic backbone: embedding → `DecoderLayer`×N → final norm → `lm_head` |
| `tensor_name.rs` | — | Scope hierarchy only (see below) |

## Tensor Names (`tensor_name.rs`)

Only the *scope hierarchy* is centralized here:

| Item | Produces |
|------|----------|
| `MODEL_SCOPE` | `"model"` |
| `layer_scope(i)` | `model.layers.{i}` |
| `token_embedding_name()` / `position_embedding_name()` / `norm_weight_name()` | `model.embed_tokens.weight` / `model.position_embedding.weight` / `model.norm.weight` |
| `lm_head_name(tie_word_embeddings)` | tied → the embedding name, else `lm_head.weight` |
| `GatedDeltaAttentionTensorNames` + `gated_delta_attention_tensor_names(i)` | the one block whose nine HF keys are declared as a struct (`model.layers.{i}.linear_attn.*`) |

Rules:

- Leaf weight names are formatted inside the module that owns them, from the scope string it is handed: `Tensor::zeros(vec![…], format!("{scope}.gate_proj.weight"))`. `DecoderLayer::new` derives `attn_scope` / `ffn_scope` from `layer_scope(i)` and passes them down — there is no per-family name object to keep in lockstep.
- Names must equal the HF safetensors keys; weight loading matches by exact name.
- Names must match `mem_pool.rs` `REGEX_SET`: `model\.layers\.\d+\.(.*)`, `.*\.weight`, `model.*\.output`, `model.*\.(q|k|v|o)_proj\.output`, etc. Non-matching names panic in `from_mem_pool`.

## Tensor API Catalog

Modules compose the graph exclusively through `Tensor` builder methods; each call allocates the output and enqueues an operator:

| File | Methods |
|------|---------|
| `tensor/ops.rs` | `add`, `silu_mul`, `sigmoid_mul`, `add_rms`, `sigmoid`, `rms`, `lookup_rms` (assoc.), `view`, `lift_vector` |
| `tensor/matmul.rs` | `matmul`, `matmul_add`, `matmul3`, `attention`, `matmul_local_topk` |
| `tensor/moe.rs` | `experts_matmul_silu_mul_matmul`, `experts_matmul_mul`, `experts_merge_add`, `softmax_norm`, `sigmoid_gate`, `topk_norm`, `topk_softmax` |
| `tensor/linear_attention.rs` | `matmul_proj`, `causal_conv_silu`, `recurrent_gated_delta_rule` |

Common arguments: `MatMulParams` (tiling, from `crate::kernel::common::matmul_params`), `decode_only_flag: bool`, and a `tensor_name: String` built from `self.scope_name`. If the needed op does not exist, create it with the `create_operator` skill first, then wire its Tensor API method.

## On Request: Fill Forward Composition

Chain Tensor API calls, naming every intermediate from the scope:

```rust
pub fn forward(&self, hidden_states: &Tensor<T>, residual: &Tensor<T>,
               decode_only_flag: bool, _tensor_name: String) -> Tensor<T> {
    let product = hidden_states.matmul(&self.weight, MatMulParams { /* tiling */ },
        hidden_states.shape[0], decode_only_flag,
        format!("{}.proj.output", self.scope_name));
    product.matmul_add(&self.down_weight, residual, MatMulParams { /* tiling */ },
        decode_only_flag, format!("{}.output", self.scope_name))
}
```

`DenseMlp::forward` is the canonical reference; `SparseMoe::forward` shows router-output (`ExpertRouting<T>`) threading. Honor `decode_only_flag` (pass it through; call `lift_vector()` on decode-side activations where the reference modules do).

## On Request: DecoderLayer Wiring

To plug a new module into the layer stack:

1. `src/config/text_config.rs`: the config layer is **data only**. If the choice is expressed in `config.json`, add the field with its raw type (`String` / `Vec<String>` / `usize` / `bool`, `#[serde(default)]` + alias when HF spells it differently), let `derive()` compute what HF computes and `validate()` normalize/whitelist it. Never add a form-selecting `enum` or a per-family config module — the only `model_type` branch allowed here is `apply_family_defaults()` filling missing numbers.
2. The layer construction point: add a variant to `AttentionBlock<T>` / `FfnBlock<T>` (`…/attention_enum.rs`, `mlp/ffn_block.rs`) or to the router enum (`mlp/sparse_moe/layer.rs`), select it with an **inline `if` on the string predicate** — `layer_types[i].contains("linear")`, the `mlp_only_layers` / `num_experts` / `decoder_sparse_step` conjunction, `scoring_func.contains("sigmoid")` — mirroring the corresponding `__init__` in Python, and dispatch it in `forward`'s `match`.
3. Inside that branch, build weight names from the scope handed to the constructor (`format!("{scope}.<hf_leaf>.weight")`); no name struct arm to extend in lockstep.
4. If the difference is an architectural fact the JSON never expresses (does this class create `q_norm`? does the norm take `(1 + w)`?), do **not** add a config field: write a `const …: &[&str]` table in the one layer file that consumes it, defaulting to true and registering the exceptions (see `NO_QK_NORM_MODELS`, `ZERO_CENTERED_MODELS`).

## On Request: Multi-File Directory Split

Follow `sparse_moe/` when a module grows variants (e.g. multiple routers):

```
src/transformer/<module>/
├── mod.rs            # private submodules; `pub use self::layer::MyModule;` (+ `#[cfg(test)] mod tests;`)
├── layer.rs          # the public module
├── <variant_a>.rs    # pub(super) helper/variant
├── <variant_b>.rs
└── tests.rs
```

Wrap variants in a private enum with `new` (dispatch on a raw `TextConfig` string field) and `forward` (match and delegate) — see `SparseMoeRouter`. Declare the directory in `src/transformer/mod.rs`.

## Unit Test Pattern

```rust
#[cfg(test)]
mod test {
    use super::*;
    use crate::runtime::SequenceSlice;
    use std::collections::HashMap;

    const EMPTY_SLICES: &[SequenceSlice] = &[];

    #[test]
    fn test_my_module() {
        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        let module = MyModule::<f32>::new(/* dims, scope string */);
        // stubbed forward: only assert weight shapes here
        // after forward lands:
        //   let output = module.forward(&input, &residual, false, String::from("test_output"));
        //   debug_assert_eq!(output.shape, expected_shape);
        //   f32::with_operator_queue(|queue| {
        //       for operator in queue.iter() {
        //           for i in 0..thread_num {
        //               operator.run(batch_size, 0, 0, batch_size, thread_num, i,
        //                            EMPTY_SLICES, &mut Vec::new());
        //           }
        //       }
        //   });
    }
}
```

Notes: initialize both globals (`init_global` + `init_operator_queue`) per dtype (`f32`/`f16`); test tensor names must also satisfy the `REGEX_SET` (use real `model.layers.N.…` prefixes — synthetic names like `g2.layers.0.self_attn` panic in `from_mem_pool`); when a test needs a config, build a `TextConfig` literal and call `TextConfig::load` only in the ckpt snapshot tests; mark model-scale tests (real `config.json` + full queue drain) with `#[ignore]` like `decoder_layer.rs` does.

## Pitfalls

- Building operators directly in `forward` — modules may only call Tensor builder methods.
- Weight names drifting from HF safetensors keys — the weight stays zeros silently (or panics with `strict_weights`).
- Intermediate tensor names outside `REGEX_SET` — `from_mem_pool` panics at graph build.
- Forgetting to thread `decode_only_flag` — prefill/decode row ranges break.
- Putting a form decision (`layer_types` / `scoring_func` / qk-norm) into `TextConfig` as an enum or a family check — the config layer stays data-only; the decision belongs to the layer construction point.
- Adding an `AttentionBlock` / `FfnBlock` / `SparseMoeRouter` variant without its `forward` match arm — the graph panics at build time.
- Storing activations in the struct — all per-round buffers live in the graph, owned by tensors.
