# eLLM：让 CPU 在长程推理任务中快过 GPU

eLLM 是一款面向纯 CPU 服务器的大模型推理框架。它采用“以存换算”策略，利用 CPU 大容量 DDR 内存，弥补其与 GPU HBM 之间的带宽差距，从而在长程任务推理场景下实现超越 GPU 的性能。Beta 版本现已发布，欢迎体验。
- **Prefill**：相较现有 CPU 推理框架可实现约**两个数量级**的性能提升
  - 长文本一次性整段 Prefill，
  - 多轮交互仅对新增输入做增量 Prefill；
- **Decode**：以更小的 batch 运行，不仅激活的参数更少，单个 request 可分得的内存带宽也更高，因此推理速度同样可以超过 GPU baseline。
👉 GitHub：https://github.com/lucienhuangfu/eLLM





# eLLM: Run Long-Horizon Inference Faster on CPUs Than on GPUs

eLLM is a Rust-based LLM inference framework for CPU-only servers. It adopts a "trade storage for computation" strategy, leveraging the CPU's large-capacity DDR memory to close the order-of-magnitude bandwidth gap against GPU HBM, and thereby delivers performance that surpasses GPUs in long-horizon inference. The Beta release is now available — you are welcome to try it out.
- **Prefill**: achieves roughly **two orders of magnitude** of performance improvement over existing CPU inference frameworks
  - full single-pass Prefill for long text;
  - incremental Prefill on only the newly added input in multi-turn interactions;
- **Decode**: runs with a smaller batch, which not only activates fewer parameters but also gives each request a larger share of memory bandwidth, so inference speed can likewise exceed GPUs.
👉 GitHub: https://github.com/lucienhuangfu/eLLM


Rethinking AI infrastructure beyond the GPU.
Building a CPU-only LLM inference framework — eLLM.
Running long-horizon inference faster on CPU than on GPU.
GitHub: https://github.com/lucienhuangfu/eLLM