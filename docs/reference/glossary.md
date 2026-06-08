# Glossary

Quick reference for terms used throughout the eLLM documentation.

---

## A

**AttentionKind**  
Enum variant describing the attention type for one decoder layer: `Full`, `SlidingWindow`, or `Linear`.

---

## B

**BatchPlan**  
The scheduling decision for one round: `Decode`, `Prefill`, or `Idle`.

**BatchScheduler**  
The component that scans `batch_list` and produces a `ScheduleTask` (prefill and decode slices) each round. Lives in `src/runtime/scheduling/scheduler.rs`.

**BatchSequence**  
Holds the tokenizer and the token sequence buffer for all batch slots. Exposes `write_prompts()` and `decode_*()` methods.

**batch_size**  
The maximum number of concurrent in-flight requests. Controlled by `ELLM_BATCH_SIZE`.

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

**FfnKind**  
Enum variant describing the feed-forward network type for one decoder layer: `Dense { intermediate_size }` or `SparseMoe { … }`.

**filling_length**  
The number of remaining prefill tokens for a batch slot. Decremented by `TopKSoftmax` after each prefill step. When it reaches 0, the slot transitions to `Phase::Decode`.

---

## G

**GQA (Grouped Query Attention)**  
An attention variant where multiple Q heads share a single K/V head. The ratio `num_attention_heads / num_key_value_heads` is `num_key_value_groups`.

---

## K

**kv_index**  
The next write position in the KV cache for a batch slot.

---

## L

**LayerPlan**  
A per-layer data structure that records `AttentionKind` and `FfnKind` for one decoder layer. Produced by the family resolver and stored in `ResolvedConfig.layers`.

---

## M

**ModelFamily**  
Enum identifying the model family: `Llama`, `Qwen`, `Mixtral`, `MiniMax`, `MiniMaxM2`, `Unknown`. Used to dispatch config resolution and tensor name generation.

---

## P

**Phase**  
The lifecycle state of a batch slot: `Start` → `Prefill` → `Decode` → `Eos` → `Start`.

**prefill round**  
A scheduling round that processes prompt tokens. A single sequence may be split across multiple prefill rounds if its length exceeds `chunk_size`.

---

## R

**ResolvedConfig**  
The stable runtime config derived from `HfConfig`. Contains no family-specific optional fields. Passed to all runtime components.

**RoPE (Rotary Position Embedding)**  
A positional encoding scheme that multiplies Q/K projections by complex rotation factors. MiniMax-M2.5 uses partial-dimension RoPE (`rotary_dim < head_dim`).

---

## S

**ScheduleTask**  
The broadcast payload sent from `BatchScheduler` to all `ServingRunner` threads. Carries `prefill_list`, `decode_list`, sizes, a timestamp, and a task ID.

**sequence_index**  
The current read/write cursor within a sequence's token buffer for a batch slot.

**SequenceSlice**  
The minimal computation unit: `batch_index`, `sequence_index`, `token_start_index`, `length`, `last_token_flag`.

**SequenceState**  
Per-slot state tracked by the serving layer: `phase`, `sequence_index`, `kv_index`, `filling_length`, `notify`.

**ServingRunner**  
The broadcast-subscribed thread-pool executor. Each runner thread subscribes to the broadcast channel and executes the operator queue when a `ScheduleTask` arrives.

---

## T

**TensorNames / ModelTensorNames**  
A per-family name description object that maps logical tensor roles (e.g., `q_proj`) to their actual safetensors key names.

**TokenCounter**  
Tracks the total tokens written since the last scheduling trigger. Fires either when the count exceeds `chunk_size` (threshold trigger) or when the timeout window expires (timeout trigger).

**TopKSoftmax**  
The final operator in the queue. Performs sampling, updates `SequenceState`, and notifies the serving layer. Transitions a slot from `Prefill` to `Decode` or from `Decode` to `Eos`.