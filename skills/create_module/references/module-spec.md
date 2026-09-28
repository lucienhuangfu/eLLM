# eLLM Transformer Module Reference

Lookup material for the `create_module` skill. The workflow lives in [SKILL.md](../SKILL.md); this file covers catalogs, wiring, and on-request extension patterns.

## Module Catalog (`src/transformer/`)

| File | Module | Role |
|------|--------|------|
| `attention.rs` | `Attention<T>` | GQA self-attention: `matmul3` (Q/K/V + qk-norm + RoPE) → `attention` → `matmul_add` (o_proj + residual) |
| `dense_mlp.rs` | `DenseMlp<T>` | SwiGLU MLP: `matmul`×2 (gate/up) → `silu_mul` → `matmul_add` (down + residual); the simplest complete example |
| `sparse_moe/` | `SparseMoe<T>` | MoE block in its own directory: `layer.rs` (public), `router_softmax.rs` / `router_sigmoid.rs` (`pub(super)` variants behind a private `SparseMoeRouter` enum), `tests.rs` |
| `decoder_layer.rs` | `DecoderLayer<T>` | Assembles one layer: input layernorm (`rms` / `lookup_rms` at layer 0) → `AttentionBlock` → post-attention layernorm → `FfnBlock` |
| `rope.rs` | `RotaryEmbedding` | Pure CPU precompute of the position-embedding table (no tensors/operators; `forward<T>() -> Vec<T>`) |
| `names.rs` | — | Tensor-name structs + `layer_tensor_names()` (see below) |

## Tensor-Name Structs (`names.rs`)

One struct per module role, all `#[derive(Debug, Clone)]` with a `scope: String` plus one field per weight:

| Struct | Fields | Generated for |
|--------|--------|---------------|
| `ModelTensorNames` | token_embedding, position_embedding, lm_head, norm_weight | whole model |
| `AttentionTensorNames` | q/k/v/o_proj, q/k_norm | `…layers.{i}.self_attn` |
| `DenseMlpTensorNames` | gate/up/down_proj | `…layers.{i}.mlp` |
| `SparseMoeTensorNames` | router_gate, router_bias (`Option<String>`), experts_gate/up/down_proj | `…layers.{i}.mlp` |
| `FfnTensorNames` | enum: `Dense(...)` \| `SparseMoe(...)` | selected by `FfnKind` |
| `LayerTensorNames` | scope, attention, ffn, input/post_attention_layernorm | per layer |

Rules when adding a new struct:

- Names must equal the HF safetensors keys, e.g. `model.layers.{i}.mlp.gate_proj.weight` — weight loading matches by exact name.
- Names must match `mem_pool.rs` `REGEX_SET`: `model\.layers\.\d+\.(.*)`, `.*\.weight`, `model.*\.output`, `model.*\.(q|k|v|o)_proj\.output`, etc. Non-matching names panic in `from_mem_pool`.
- The enum wrapper pattern (`FfnTensorNames`) keeps `DecoderLayer::new` matching `FfnKind` and names in one `match` — extend both arms together.

## Tensor API Catalog

Modules compose the graph exclusively through `Tensor` builder methods; each call allocates the output and enqueues an operator:

| File | Methods |
|------|---------|
| `tensor/ops.rs` | `add`, `silu_mul`, `add_rms`, `sigmoid`, `rms`, `lookup_rms` (assoc.), `lift_vector` |
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

1. `model_family/config`: extend `AttentionKind` (`attention_kind.rs`) or `FfnKind` (`ffn_kind.rs`) with the new variant (+ parsing in `for_layer` if it is config-driven); re-export in `config/mod.rs`.
2. `decoder_layer.rs`: add a variant to `AttentionBlock<T>` / `FfnBlock<T>`, construct it in `DecoderLayer::new`'s `match` on `config.layers[layer_idx].attention` / `.ffn` (names come from `layer_tensor_names`), and dispatch it in `forward`'s `match`.
3. `names.rs`: generate the module's name struct inside `layer_tensor_names()` for the matching `FfnKind`/`AttentionKind` arm — the `(kind, names)` match in `DecoderLayer::new` requires them to stay in lockstep.

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

Wrap variants in a private enum with `new` (dispatch on a `*Kind` config) and `forward` (match and delegate) — see `SparseMoeRouter`. Declare the directory in `src/transformer/mod.rs`.

## Unit Test Pattern

```rust
#[cfg(test)]
mod test {
    use super::*;
    use crate::model_family::config::Config;
    use crate::runtime::SequenceSlice;
    use std::collections::HashMap;

    const EMPTY_SLICES: &[SequenceSlice] = &[];

    #[test]
    fn test_my_module() {
        f32::init_global(HashMap::new());
        f32::init_operator_queue();

        let module = MyModule::<f32>::new(/* dims, names */);
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

Notes: initialize both globals (`init_global` + `init_operator_queue`) per dtype (`f32`/`f16`); test tensor names must also satisfy the `REGEX_SET`; mark model-scale tests (real `config.json` + full queue drain) with `#[ignore]` like `decoder_layer.rs` does.

## Pitfalls

- Building operators directly in `forward` — modules may only call Tensor builder methods.
- Weight names drifting from HF safetensors keys — the weight stays zeros silently (or panics with `strict_weights`).
- Intermediate tensor names outside `REGEX_SET` — `from_mem_pool` panics at graph build.
- Forgetting to thread `decode_only_flag` — prefill/decode row ranges break.
- Adding a `FfnKind` variant without the matching `FfnTensorNames` arm — `DecoderLayer::new` hits `unreachable!`.
- Storing activations in the struct — all per-round buffers live in the graph, owned by tensors.
