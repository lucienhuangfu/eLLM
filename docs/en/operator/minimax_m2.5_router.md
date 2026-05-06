# MiniMax M2.5 Router Design

## 1. Document Goal

This document explains the router design of `MiniMax-M2.5` and aligns it with the current implementation structure.

The goal is not to paste code, but to answer three questions:

1. Why this router should not be understood as a normal softmax MoE router
2. Which `operators` are merged together in the current implementation
3. How those merged operators form a complete data flow

---

## 2. Router Goals

The `MiniMax-M2.5` router must satisfy these constraints:

1. `num_local_experts = 256`
2. `num_experts_per_tok = 8`
3. `scoring_func = sigmoid`
4. `use_routing_bias = true`
5. `gate` does not participate in fp8 conversion

This means it is not an ordinary `softmax top-k` router. It is a gate routing mechanism with an independent scoring semantic and routing bias.

So this document defines it as:

> `MiniMaxM2RouterV1 = gate linear + sigmoid scoring + routing bias + top-k dispatch`

---

## 3. Overall Structure

The current implementation does not perform the whole router pipeline in one operator. Instead, it is split into two parts:

1. `ExpertsSigmoidGate` generates expert scores
2. `ExpertsTopkNorm` converts the scores into the final routing result

So the router pipeline can be summarized as:

`hidden_states -> gate scoring -> top-k norm -> dispatch buffers`

This is the most important boundary in the current design.

---

## 4. How the Operators Are Merged

The core of this code change is that the original scattered gate logic has been collected into a separate operator.

### 4.1 What Was Merged

`ExpertsSigmoidGate` now handles not just a single matrix multiplication, but a combination of three things:

1. Linear projection
2. Routing bias injection
3. Sigmoid activation

So the `gate` step is no longer "call matmul and then patch in extra logic." It has become a complete gating operator.

### 4.2 What Was Not Merged

Some capabilities remain separate:

1. `MatMul` is still the provider of the underlying block multiplication capability
2. `ExpertsTopkNorm` still handles top-k selection and normalization
3. Dispatch buffer organization is still maintained by the router upper layer

So "merge" here does not mean cramming everything into one huge function. It means turning the tightest part of the logic into a semantically complete operator.

### 4.3 Why the Merge Is Useful

This split gives three direct benefits:

1. The semantics of `gate` are clearer and no longer depend on a generic `MatMul` container
2. `routing bias` is no longer a post-processing patch; it is part of the compute chain
3. The kernel only needs to focus on gate semantics and does not need to know the full router shape above it

---

## 5. Computation Flow

### 5.1 Gate Projection

`hidden_states` first enter the gate projection and produce raw scores for each token over all local experts.

These scores are not the output of a normal classification head. They are the router entry scores.

### 5.2 Sigmoid Scoring

The current design uses `sigmoid` as the router's main semantic.

It expresses the independent activation tendency of each expert, rather than the global competition-style probability used by softmax.

### 5.3 Routing Bias

`use_routing_bias = true` means the bias must participate during scoring, not be added after top-k.

Only then can the bias truly affect expert ranking and selection.

### 5.4 Top-k Selection

`num_experts_per_tok = 8` determines that each token must keep 8 experts.

So the later `ExpertsTopkNorm` is responsible for:

1. Selecting the top-8 experts
2. Producing the corresponding weights
3. Producing the index and indicator information

---

## 6. Responsibilities at the Code Level

### 6.1 `ExpertsSigmoidGate`

This operator handles the gate computation itself and already includes:

1. Input and weight pointers
2. Output buffer
3. Tile scheduling parameters
4. Thread-private panel pools

Its job is to compute the gate step completely, not just call a normal matmul once.

### 6.2 `MatMulTrait`

`MatMulTrait` remains the low-level multiplication interface.

Its role is not to carry router semantics, but to provide reusable block multiplication capability and a unified computation entry point for the gate.

### 6.3 `ExpertsSigmoidGateTrait`

This trait attaches gate semantics to the multiplication path.

Its purpose is to say: this is not a normal matrix multiplication, but a gate computation with sigmoid router semantics.

### 6.4 `ExpertsTopkNorm`

This operator handles the post-gate cleanup work:

1. Top-k selection
2. Weight normalization
3. Expert indicator and index output

It is chained after `ExpertsSigmoidGate`, not a replacement for it.

---

## 7. Data-Flow Contract

### Input

The router input mainly has two kinds:

1. `hidden_states`
2. Gate weights and routing configuration

`hidden_states` are token-level representations, and the router uses them to compute each token's preference over experts.

### Output

The router ultimately needs to provide MoE dispatch information:

1. Token-to-expert selection results
2. Token-to-expert routing weights
3. Indices needed for token dispatch and recovery
4. Normalization information corresponding to top-k

In the current implementation, these are completed in two steps by `ExpertsSigmoidGate` and `ExpertsTopkNorm`.

---

## 8. Configuration Constraints

The following configuration values are the key boundaries of this router version:

| Parameter | Value | Meaning |
| --- | --- | --- |
| `hidden_size` | 3072 | Router input dimension |
| `num_hidden_layers` | 62 | Router appears in all layers |
| `num_local_experts` | 256 | Total number of local experts |
| `num_experts_per_tok` | 8 | Number of experts selected per token |
| `scoring_func` | `sigmoid` | Router scoring semantic |
| `use_routing_bias` | `true` | Bias participates in routing |
| `tie_word_embeddings` | `false` | Not directly related to the router |
| `qk_norm_type` | `per_layer` | Attention-related, not a core router parameter |

The most important pieces are:

1. `num_local_experts = 256`
2. `num_experts_per_tok = 8`
3. `scoring_func = sigmoid`

These three define the router version boundary.

---

## 9. Quantization Constraints

The configuration also includes an important note:

`modules_to_not_convert = ["gate", "e_score_correction_bias", "lm_head"]`

This means the router-related modules should not go through the normal quantization conversion path.

In documentation terms, it means:

1. The gate stays high precision
2. The routing bias stays high precision
3. Router numerical stability is prioritized over compression ratio

---

## 10. Recommended Naming

To keep code and documentation consistent, the following naming scheme is recommended:

1. `RouterV1`: `sigmoid + topk + routing bias`
2. `MiniMaxM2RouterV1`: dedicated MiniMax M2.5 version
3. `SparseMoeRouterV1`: generic softmax router version

If the design continues to evolve, you can add:

1. `MiniMaxM2RouterV2`
2. `MiniMaxM25RouterV1`

---

## 11. Conclusion

The `MiniMax-M2.5` router is better understood as an independent version rather than a simple parameterization of `SparseMoeRouter`.

The key points of the current implementation can be summarized as:

1. `gate` already merges `matmul + routing bias + sigmoid`
2. The upper router layer still keeps the two-stage `gate -> topk` structure
3. `ExpertsTopkNorm` translates scores into the final dispatch result

If we keep only one sentence:

> `MiniMaxM2RouterV1 = gate linear + sigmoid scoring + routing bias + top-k dispatch`
