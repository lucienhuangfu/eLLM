# Design

Internal architecture documentation for the eLLM inference engine.

## Runtime

- [Runtime Overview](runtime/overview.md) — module structure, component map, data flow
- [Scheduler Design](runtime/schedule.md) — BatchScheduler, SliceScheduler, event-driven triggering

## Transformer

- [Minimal Model Abstraction](transformers/minimal_model_abstraction.md) — ModelFamily, LayerPlan, TensorNames
- [MiniMax-M2.5 RoPE](transformers/minimax_m2.5_rope.md) — partial-dimension RoPE, attention_scaling
- [MoE Routing Data Structures](transformers/moe_routing_data_structures.md) — compact expert queues, routing write flow

## Operators

- [Attention](operators/attention.md) — CPU static parallel allocation, GQA, causal semantics
- [MatMul](operators/matmul.md) — blocking, packing, SIMD/FMA, multithreaded tiling
- [LiftVector](operators/left_vector.md) — decode buffer compaction, last-token extraction
- [GlobalIndexLookup](operators/global_index_lookup.md) — reverse mapping from global token index to sequence position
- [MiniMax-M2.5 Router](operators/minimax_m2.5_router.md) — ExpertsSigmoidGate, ExpertsTopkNorm, routing pipeline
- [FakeEcho](operators/fake_echo.md) — test operator for integration debugging