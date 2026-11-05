# Model Abstraction: Two Planes, No Relay

## Status

This document describes the **shipped** config-to-model path.

It supersedes the earlier proposal of the same file, which added three relay
objects between `config.json` and the layer stack:

| Retired | Replaced by |
|---|---|
| `ModelFamily` / `ModelName` enum | the raw `TextConfig::model_type` string |
| `LayerPlan` + `AttentionKind` / `FfnKind` enums | inline string predicates at the layer construction point |
| `RouterScoringKind` (`router_scoring.rs`) | `scoring_func: String` + `contains("sigmoid")` in `mlp/sparse_moe/layer.rs` |
| `LayerSpec` (`layer_spec.rs`) | flat `Vec<String> layer_types` on `TextConfig` |
| `HfConfig` + a model-side `ResolvedConfig` two-layer split | one `TextConfig` (a 1:1 mirror of `config.json`) + its `load()` pipeline. The `ResolvedConfig` that still exists in `src/config/config_types.rs` is the *launch* config (CLI + YAML), not an architecture config |
| `src/model_spec/`, `src/model_family/` | deleted; the vision config lives in `src/config/vision_config.rs` |
| `norm_topk_prob` | dropped entirely: the router always normalizes inside the selected top-k |
| `TensorNames` per-family name objects | `src/transformer/tensor_name.rs`, scope hierarchy only |

The proposal's diagnosis was right — the problem was coupling, not directory
depth — but its cure kept a blueprint layer in the middle. That layer is now
gone.

## The two planes

Everything about a model is split by *kind of question*, not by family:

```
config.json (flat, or multimodal with a nested text_config)
   |  (1) serde read: field name == JSON key, struct-level #[serde(default)],
   |      missing key == sentinel 0 / false / "" / empty Vec
   v
TextConfig                      <- DATA PLANE: src/config/text_config.rs
   |  (2) apply_family_defaults()  class-B numeric defaults, the ONLY model_type match
   |      derive()                  branch-free arithmetic
   |      validate()               normalize + whitelist + load-time hard failure
   v
TextModel::new(&config) -> DecoderLayer::new(&config, i)      <- JUDGMENT PLANE
   |- config.layer_types[i].contains("linear")  -> GatedDelta | Full
   |- mlp_only_layers / num_experts / decoder_sparse_step -> Dense | SparseMoe
   |- config.scoring_func.contains("sigmoid")   -> Sigmoid | Softmax router
   |- full_attention.rs: NO_QK_NORM_MODELS      -> create q_norm/k_norm or not
   '- decoder_layer.rs:  ZERO_CENTERED_MODELS   -> norm takes w or (1 + w)
```

- **Data plane** knows only facts that a `config.json` can state. It never picks
  a layer shape.
- **Judgment plane** is `src/transformer/**`, i.e. the code of the layer that is
  actually built. This mirrors Python, where the same decisions sit in the class
  body and `__init__` of `modeling_<name>.py` — not in the config object.

## Judgment classes A / B / C

The rule that decides where a difference goes:

| Class | Question | Example | Lives in |
|---|---|---|---|
| **A** | per-layer *form*, read from data | attention linear vs full; FFN dense vs MoE; router softmax vs sigmoid | inline `if` at the construction point (`attention/attention_enum.rs`, `mlp/ffn_block.rs`, `mlp/sparse_moe/layer.rs`) |
| **B** | whole-model *missing numbers* | `full_attention_interval = 4`, `partial_rotary_factor = 0.25`, the five GatedDeltaNet dims | the single `matches!(model_type, …)` in `TextConfig::apply_family_defaults()` |
| **C** | architectural fact the JSON never expresses | does this class create `q_norm`? does RMSNorm take `(1 + w)`? | a `const &[&str]` table inside the layer file that consumes it, **defaulting to true, exceptions registered** |

B vs C is the key distinction: B is "the JSON forgot a number" (a config
responsibility); C is "does the constructor contain that line" (never a config
responsibility).

Class A, verbatim from the code:

```rust
// src/transformer/attention/attention_enum.rs
let is_linear = config.layer_types.get(layer_idx).is_some_and(|t| t.contains("linear"));

// src/transformer/mlp/ffn_block.rs
let sparse = !config.mlp_only_layers.contains(&layer_idx)
    && config.num_experts > 0
    && (layer_idx + 1) % config.decoder_sparse_step == 0;

// src/transformer/mlp/sparse_moe/layer.rs
if scoring.contains("sigmoid") { /* sigmoid router */ } else { /* softmax router */ }
```

Class C:

```rust
// src/transformer/attention/full_attention.rs
pub(crate) const NO_QK_NORM_MODELS: &[&str] = &["llama", "mixtral", "qwen2", "qwen2_moe"];
pub(crate) fn use_qk_norm(model_type: &str) -> bool { !NO_QK_NORM_MODELS.contains(&model_type) }

// src/transformer/decoder_layer.rs
pub(crate) const ZERO_CENTERED_MODELS: &[&str] = &[];   // pending alignment proof (gap G4)
```

`use_qk_norm == false` must also *not register* the two weight names, because
strict weight loading panics on a name the checkpoint does not have; hence
`q_norm_weight` / `k_norm_weight` are `Option<Tensor<T>>` and the fused
`matmul3` never dereferences them when the flag is off.

## Config rules

- **Raw types, sentinel defaults.** `layer_types: Vec<String>`,
  `scoring_func: String`, `num_experts: usize`. `0` / `""` / `false` / empty
  `Vec` mean "not given or not applicable", which is what the old
  `Option` + `unwrap_or` chain meant.
- **The `null` red line.** `#[serde(default)]` covers a *missing* key, not a key
  whose value is `null`. Verified against every `checkpoints/*/config.json`, only
  three keys are ever explicitly `null` and they must stay `Option`:
  `rope_scaling`, `pad_token_id`, `sliding_window`.
- **Aliases, not extra fields.** Where two HF spellings mean the same number,
  one field carries an alias: `num_local_experts` → `num_experts`,
  `rope_parameters` → `rope_scaling`. Where the old spelling was simply wrong
  (`attention_bias`, never `qkv_bias`, in every real checkpoint) the key is
  renamed outright and no alias is kept — the wrong name must not survive the
  grep sweep.
- **Neutrality by arithmetic.** The trick that lets two `Option`s disappear:
  `full_attention_interval` defaults to `1`, so `(i + 1) % 1 == 0` is always
  true and `derive()` synthesizes an all-`full_attention` list — byte-for-byte
  the old "no layer_types means full attention everywhere" behavior, with no
  special branch. `qwen3_5` overrides it to `4` in class B.
- **Fail at load, not mid-inference.** `validate()` lowercases and trims,
  asserts `layer_types.len() == num_hidden_layers`, whitelists the
  `linear` / `full` and `softmax` / `sigmoid` vocabularies. A short list used to
  fall through `.get(i) -> None` into a silent Full default — a bug magnet.
- **Multimodal is one read.** `RootConfig { #[serde(flatten)] text, text_config:
  Option<Box<TextConfig>>, vision_config, four vision token ids }`, with the
  nested `text_config` taking precedence — the only place "nested wins" is
  expressed. Both the top-level `qwen3_5` and the nested `qwen3_5_text` spellings
  are listed in every `matches!`; there is no normalization function.

## Naming

`src/transformer/tensor_name.rs` centralizes the *scope hierarchy* only
(`MODEL_SCOPE`, `layer_scope(i)`, the model-level weight names,
`lm_head_name(tie_word_embeddings)`). Leaf names are formatted inside the
operator that owns them, from the scope string it is handed, so the config layer
keeps no dependency on any concrete model and the name lives next to its only
consumer. Names still have to equal the HF safetensors keys and match
`mem_pool.rs`'s `REGEX_SET`, which is what routes a tensor to a memory pool —
synthetic prefixes panic at graph build.

## Trade-offs accepted

- **No exhaustive match.** Five string comparisons with fallbacks replace five
  enums. The repository had already taken this step for `rope_type` and
  `hidden_act`; the compensation is front-loading legality into `validate()`.
- **Class B stays in the config layer, a deliberate deviation from Python.**
  Python spreads defaults over `configuration_<name>.py`; Rust has no per-family
  config file, so one ~35-line `matches!` is its closest equivalent. Spreading it
  across layer files would be more fragmented than Python, not less.
- **Class C tables are family knowledge kept by comment.** Each table has 0-1
  entries, sits in the file that consumes it, and points back at the Python class
  it corresponds to. If a family ever differs by more than one bool, per-family
  layer classes become worth revisiting — not before.
- **No `deny_unknown_fields`.** A misspelled key silently takes its default (gap
  G1 was exactly this). The G5 inventory comment at the top of `text_config.rs`
  is the only guard for now.

## Adding a model

1. Drop the checkpoint under `checkpoints/` and add a snapshot test that calls
   `TextConfig::load` and locks `(layer count, layer_types, per-layer sparse
   predicate, head_dim, rotary_dim, num_experts, scoring_func)`.
2. If the JSON spells an existing concept differently, add a `#[serde(alias)]`.
3. If it omits numbers the layers need, add a class-B branch in
   `apply_family_defaults()`.
4. If it introduces a new per-layer form, extend the inline predicate at the
   construction point — not an enum in the config.
5. If its class body creates (or omits) a submodule that the shared layer code
   assumes, register the `model_type` in the class-C table of that layer file.

## Known gaps

| Ref | Item | State |
|---|---|---|
| G0 | `num_local_experts` (MiniMax's 256 experts) | fixed by serde alias |
| G1 | the projection-based key name was wrong; HF uses `attention_bias` in every checkpoint | fixed by renaming the field and the key; the old spelling gets no alias, since it exists in no real `config.json` |
| G2 | `attn_output_gate` / `output_gate_type` | implemented: `q_proj` doubles its rows and `matmul3` emits the gate half per head; the gate is fused into the `attention` operator's col-loop epilogue, which multiplies the attention output in place by `sigmoid(gate)`, matching the reference `modeling_qwen3_5.py` L808 (`attn_output * torch.sigmoid(gate)`). Note: `output_gate_type` is parsed and asserted to be `"swish"`, but the reference forward still uses plain sigmoid for this gate (not `ACT2FN`), so the epilogue uses `sigmoid(gate)`. |
| G3 | `hf_config_path` was ignored by `load` | fixed: an explicit path replaces `{model_dir}/config.json` |
| G4 | zero-centered RMSNorm for `qwen3_5` | not enabled; `ZERO_CENTERED_MODELS` stays empty until the alignment run proves it |
| G5 | keys intentionally not consumed (`mtp_*`, `mamba_ssm_dtype`, `attention_dropout`, `initializer_range`, `qk_norm_type`, `shared_intermediate_size`, …) | inventoried in the module doc of `text_config.rs`, each marked "not needed for inference" or "to wire" |
