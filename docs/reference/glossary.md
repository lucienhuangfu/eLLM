# Glossary

Quick reference for terms used throughout the eLLM documentation.

---

## A

**`attn_output_gate`**  
HF config flag that doubles the `q_proj` row count so every attention head also emits a gate; the attention output is multiplied by `sigmoid(gate)`, matching the reference `modeling_qwen3_5.py` L808 (`attn_output * torch.sigmoid(gate)`). The gate rows are produced by `matmul3` (interleaved with the query rows per head) and the gating is fused into the `attention` operator's col-loop epilogue (`src/operators/attention/full_attention`), which applies `attn_output *= sigmoid(gate)` in place; `src/transformer/attention/full_attention.rs` wires it. There is no longer a standalone gate node in the graph. (`output_gate_type` is parsed and asserted to be `"swish"`, but the reference forward uses plain sigmoid for this gate, not `ACT2FN`.)

**Attention dispatch**  
Per-layer attention form is a plain string test at the construction point — `config.layer_types[i].contains("linear")` in `src/transformer/attention/attention_enum.rs`. The former `AttentionKind` enum blueprint is retired.

---

## B

**BatchPlan**  
The scheduling decision for one round: `Decode`, `Prefill`, or `Idle`.

**BatchScheduler**  
The component that scans `batch_list` and produces a `ScheduleTask` (prefill and decode slices) each round. Lives in `src/runtime/scheduling/scheduler.rs`.

**BatchSequence**  
Holds the tokenizer and the token sequence buffer for all batch slots. Exposes `write_prompts()` and `decode_*()` methods.

**batch_size**  
The maximum number of concurrent in-flight requests. Controlled by `ELLM_BATCH`.

---

## C

**chunk_size**  
The maximum number of tokens processed in a single prefill round. Also used as the `token_threshold` for the `TokenCounter`. Controlled by `ELLM_CHUNK_SIZE`.

---

## D

**DecodeList**  
A list of `SequenceSlice`s for the decode round. Each slice has length 1 in decode mode. Supports O(log N) lookup by global token index.

**decode round**  
A scheduling round where every active `Phase::Decode` sequence contributes exactly one token to the computation.

---

## F

**`FfnBlock`**  
Runtime enum in `src/transformer/mlp/ffn_block.rs`: `Dense(DenseMlp)` or `SparseMoe(SparseMoe)`. The choice is the inline three-conjunction predicate `!mlp_only_layers.contains(i) && num_experts > 0 && (i + 1) % decoder_sparse_step == 0`, read straight from `TextConfig`; the former `FfnKind` enum is retired.

**filling_length**  
The number of remaining prefill tokens for a batch slot. Decremented by `TopKSoftmax` after each prefill step. When it reaches 0, the slot transitions to `Phase::Decode`.

---

## G

**GQA (Grouped Query Attention)**  
An attention variant where multiple Q heads share a single K/V head. The ratio `num_attention_heads / num_key_value_heads` is `num_key_value_groups`.

---

## J

**Judgment classes (A / B / C)**  
The three-way split that decides where a config-driven difference belongs. *A — per-layer form*: an inline `if` at the construction point in `src/transformer/**`. *B — data defaults for a whole model*: the single `matches!(model_type, …)` block in `TextConfig::apply_family_defaults`. *C — architectural facts absent from `config.json`* (qk-norm, zero-centered RMSNorm): a constant table inside the layer file that consumes it, defaulting to true with exceptions registered.

---

## K

**kv_index**  
The next write position in the KV cache for a batch slot.

---

## L

**`layer_types`**  
Per-layer attention form as a plain `Vec<String>` mirrored from `config.json`. When the key is absent, `TextConfig::derive` synthesizes it from `full_attention_interval`; consumers test it with `contains("linear")` instead of unpacking a blueprint object. Replaces the former `LayerPlan`.

**Load-time validation**  
`TextConfig::validate` normalizes and checks `layer_types` / `scoring_func` and asserts required dimensions while the config is read, so malformed JSON panics at load instead of silently defaulting mid-inference.

---

## M

**Missing-key sentinels**  
The zero values `#[serde(default)]` produces for absent JSON keys — `0`, `false`, `""`, empty `Vec` — which the config layer reads as "not given / feature not present" (`num_experts == 0` means no MoE, `head_dim == 0` means "derive it"). Keys that can legitimately hold an explicit `null` (`rope_scaling`, `pad_token_id`, `sliding_window`) must stay `Option`.

---

## P

**Phase**  
The lifecycle state of a batch slot: `Start` → `Prefill` → `Decode` → `Eos` → `Start`.

**prefill round**  
A scheduling round that processes prompt tokens. A single sequence may be split across multiple prefill rounds if its length exceeds `chunk_size`.

---

## R

**RoPE (Rotary Position Embedding)**  
A positional encoding scheme that multiplies Q/K projections by complex rotation factors. MiniMax-M2.5 uses partial-dimension RoPE (`rotary_dim < head_dim`).

**`RootConfig`**  
Serde view of a (possibly multimodal) `config.json`: text fields are flattened at the top level, a nested `text_config` takes precedence when present, and `vision_config` plus the vision token ids are captured alongside. Produced only inside `TextConfig::load`.

**ResolvedConfig**  
The *launch* config in `src/config/config_types.rs`: CLI flags + YAML/JSON args merged and validated (`config_validator.rs`), holding model path, batch/chunk sizes, thread counts, runner type. Unrelated to model architecture — the model-side mirror is `TextConfig`, and the former architecture `ResolvedConfig` derived from `HfConfig` is retired.

---

## S

**`scoring_func`**  
Router scoring as a raw string from `config.json` (`softmax` / `sigmoid`), normalized in `TextConfig::validate` and dispatched with `contains("sigmoid")` in `src/transformer/mlp/sparse_moe/layer.rs`. The former `RouterScoringKind` enum is retired.

**ScheduleTask**  
The payload for batch execution tasks. Carries `prefill_list`, `decode_list`, sizes, and a task ID. Used by ExecutorPool for manual batch execution.

**sequence_index**  
The current read/write cursor within a sequence's token buffer for a batch slot.

**SequenceSlice**  
The minimal computation unit: `batch_index`, `sequence_index`, `token_start_index`, `length`, `last_token_flag`.

**SequenceState**  
Per-slot state tracked by the serving layer: `phase`, `sequence_index`, `kv_index`, `filling_length`, `notify`.

**ExecutorPool**  
The leader-follower thread-pool executor. Leader thread waits for requests and builds execution plans, follower threads execute the operator queue based on shared state.

---

## T

**`tensor_name.rs`**  
`src/transformer/tensor_name.rs` centralizes only the *scope* hierarchy (`MODEL_SCOPE`, `layer_scope(i)`, model-level weight names). Leaf weight names are formatted inside the operator that owns them, from the scope it is handed, so `config` keeps no dependency on a concrete model family.

**TextConfig**  
The single text-model config struct (`src/config/text_config.rs`), a 1:1 mirror of `config.json`. `TextConfig::load(path)` = serde read → `apply_family_defaults` (class B) → `derive` (pure arithmetic) → `validate` (load-time hard failure). Replaces the former `HfConfig` / `ResolvedConfig` / `LayerSpec` / `model_family::*` chain.

**TokenCounter**  
Tracks the total tokens written since the last scheduling trigger. Fires either when the count exceeds `chunk_size` (threshold trigger) or when the timeout window expires (timeout trigger).

**TopKSoftmax**  
The final operator in the queue. Performs sampling, updates `SequenceState`, and notifies the serving layer. Transitions a slot from `Prefill` to `Decode` or from `Decode` to `Eos`.
