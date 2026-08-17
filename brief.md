# eLLM：让 CPU 在长程推理任务中快过 GPU

eLLM 是一款面向纯 CPU 服务器的大模型推理框架。它采用“以存换算”策略，利用 CPU 大容量 DDR 内存，弥补其与 GPU HBM 之间的带宽差距，从而在长程任务推理场景下实现超越 GPU 的性能。当前版本已完成核心功能开发，推理结果与现有框架完全对齐，Beta 版本现已发布，欢迎体验。

- 在 Prefill 阶段，相较于现有 CPU 推理框架，eLLM 可实现约**两个数量级**的性能提升：
  - 支持超长文本一次性完成 Prefill，避免分块处理带来的额外开销与重复计算；
  - 缓存多轮交互中的 KV Cache，仅针对新增输入执行增量 Prefill。
- 在 Decode 阶段，主要开销来自模型参数和 KV Cache 的加载，其中 KV Cache 占比较高。eLLM 采用更小的 batch 运行，不仅能够减少激活参数量，还能为单个 request 分配更高的内存带宽载入 KV Cache，因此其推理速度同样可以超过 GPU baseline。
👉 项目主页：[https://github.com/lucienhuangfu/eLLM](https://github.com/lucienhuangfu/eLLM) 
