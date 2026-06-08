# 设计

eLLM 推理引擎的内部架构文档。

## Runtime 层

- [Runtime 模块总览](runtime/overview.md) — 模块结构、组件关系、数据流
- [调度器设计详解](runtime/schedule.md) — BatchScheduler、SliceScheduler、事件驱动触发

## Transformer 层

- [最小模型抽象](transformers/minimal_model_abstraction.md) — ModelFamily、LayerPlan、TensorNames
- [MiniMax-M2.5 RoPE 原理](transformers/minimax_m2.5_rope.md) — 部分维度 RoPE、attention_scaling
- [MoE Routing 数据结构调整](transformers/moe_routing_data_structures.md) — 紧凑 Expert 队列、路由写入流程

## 算子层

- [Attention](operators/attention.md) — CPU 静态并行分配、GQA、因果语义
- [MatMul](operators/matmul.md) — 分块、packing、SIMD/FMA、多线程 tiling
- [LiftVector](operators/left_vector.md) — Decode 缓冲区压缩、末 token 提取
- [GlobalIndexLookup](operators/global_index_lookup.md) — 全局 token 索引到序列位置的反向映射
- [MiniMax-M2.5 Router](operators/minimax_m2.5_router.md) — ExpertsSigmoidGate、ExpertsTopkNorm、路由流水线
- [FakeEcho](operators/fake_echo.md) — 用于集成调试的测试算子
