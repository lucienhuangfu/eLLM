# eLLM：让 CPU 在长程推理任务中快过 GPU

eLLM 是一款面向纯 CPU 服务器的大模型推理框架。它采用“以存换算”策略，利用 CPU 大容量 DDR 内存，弥补其与 GPU HBM 之间的带宽差距，从而在长程任务推理场景下实现超越 GPU 的性能。Beta 版本现已发布，欢迎体验。

- **Prefill**：相较现有 CPU 推理框架可实现约**两个数量级**的性能提升
  - 长文本一次性整段 Prefill，
  - 多轮交互仅对新增输入做增量 Prefill；
- **Decode**：以更小的 batch 运行，不仅激活的参数更少，单个 request 可分得的内存带宽也更高，因此推理速度同样可以超过 GPU baseline。
👉 项目主页：[https://github.com/lucienhuangfu/eLLM](https://github.com/lucienhuangfu/eLLM) 
