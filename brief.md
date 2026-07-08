# eLLM：让 CPU 在长程推理中超越 GPU
eLLM 是面向 CPU-only 服务器的大模型推理框架。当前版本已完成核心功能开发，推理结果与现有框架完全对齐，Beta 版本已发布，欢迎体验。
- 在长文本场景下，Prefill 和 Decode 性能可超过 GPU baseline。
- 在多轮交互场景下，整体任务完成时间（TTC）可超过 GPU baseline。

## 核心优化

- 利用 CPU 大内存, 支持超长文本一次性 Prefill，避免分块处理和重复计算。
  利用 CPU 大内存，缓存多轮交互中的所有 KV Cache，并增量 Prefill 新输入。
- 利用 CPU 大 Cache：采用逐头 Attention 计算，减少 KV Cache 的重复内存访问。
