# eLLM：让 CPU 在长程推理中跑赢 GPU
## eLLM：让 CPU 成为 AI 推理芯片的首选
👉 项目主页：[https://github.com/lucienhuangfu/eLLM](https://github.com/lucienhuangfu/eLLM)  
🌐 语言版本：[English](README.md) | [简体中文](README.zh-CN.md)  
📚 文档：[Documentation](docs/index.md)  
🎓 目前仅开放 1–2 个 Trainee 名额，欢迎计算机专业在校生报名  
💼 我们致力于推动开源与 AI 民主化，期待与产业携手合作  
📧 联系方式：**lucienhuangfu@outlook.com**

## 🚀 进展与更新
- `v0.1.0`（2026-07-10）：发布 Beta 版本
- `v0.0.2`（2026-04-06）：发布 Alpha 版本
- `v0.0.1`（2025-12-20）：项目开源

## 🔑 功能
**eLLM**：面向 CPU 服务器的大模型推理框架
- 纯 CPU 推理，无需 GPU / NPU
- 兼容 vLLM API，可直接接入现有生态
- 推理结果与 GPU 保持一致

## 🖥️ 硬件要求
- CPU：Intel Xeon / AMD EPYC（推荐 Xeon Gen4+）
- 内存：足量 DDR（按模型规模配置）
  
## ✨ 优势
eLLM 充分释放了 **CPU 在长程推理场景下的体系结构优势**，使其在多项关键指标上实现对 GPU 推理的全面超越：
- **低延迟**：整段 Prefill，增量 Prefill，显著降低首 token 延迟
- **高吞吐**：单实例并发度虽低，但由于端到端延迟更小，**实际 QPS 反而更高**
- **长上下文**：大内存支持近乎无限长度的上下文窗口
- **低能耗**：Prefill 阶段仅加载一次参数，大幅降低重复访存的能耗
- **低成本**：硬件成本与单用户推理成本显著低于 GPU 方案

## 应用场景
eLLM 适合**长程任务**，即需要在长时间、多步骤执行过程中持续保持目标、状态与推理一致性的 Agent 工作流：
- **Open Claw（Computer-use Agent）**
  - 在执行过程中按需动态加载技能与上下文，持续规划、调用工具并完成复杂任务
  - 属于在线交互式 Agent 场景，强调低延迟、高频工具调用，以及长时间保持任务目标与执行状态
- **Code Copilot**
  - 面向跨文件、跨模块的大型代码仓库，支持长时间的软件工程任务
  - 能够在多轮编辑、测试、调试和修复过程中持续维护项目状态，适用于代码重构、Bug 修复、代码审查等 Agent Coding 场景
- **RAG（Retrieval-Augmented Generation）**
  - 在任务执行过程中动态检索并注入外部知识，而非一次性加载全部上下文
  - 适用于企业知识库问答、长文档分析以及需要持续检索与推理的 Agent 工作流
- **Deep Research**
  - 支持多轮检索、信息整合、规划与推理，在长周期研究过程中持续积累和利用中间结果
  - 适用于持续数小时甚至数天的研究任务，而不仅仅是单次长上下文推理

## ⚙️ 方法
为了更好地支持**长程任务（Long-Horizon Tasks）**，eLLM 针对“多步骤执行 + 长时间状态维护 + 低延迟交互”的 Agent 场景，基于 CPU 体系结构（内存大、缓存大、算力相对弱）提出“以存换算”的整体设计理念，将推理过程重构为**可复用、可持续增长、可局部增量更新的执行链路**，从而降低长任务中的重复计算与状态重建开销

- 🧩 **弹性静态计算图**
  构建全局唯一的静态计算图，并采用 **维度优先（dimension-first）** 的布局存取张量，让相同逻辑坐标的元素稳定映射到相同内存位置，使同一套执行图可以在不重建计算图的前提下支持不同输入长度。
- 🧊 **静态形状 KV Cache（不分页）**
  为 KV Cache 预分配固定形状的 tensor，不依赖分页式 block 管理；读写时直接按张量坐标定位 KV，并沿 sequence 维度连续读取 KV，减少元数据维护、地址映射和动态分配开销，尽量避免 TLB miss 和 cache miss。
- 📦 **超大维度张量**
  为张量预留足够大的 token/sequence 维度，构建近似“无限长度”的 KV tensor，支持整段 Prefill，从而尽量避免重复 Prefill 和参数反复载入，适配超长 Prompt 和长上下文。
- 🔁 **Session Cache（跨轮状态保持）**
  在多轮交互中持续保留 KV 状态，仅对新输入进行增量 Prefill，而无需重复计算历史上下文；从机制上实现“状态连续性”，支撑 Agent 在长时间任务中保持上下文一致与执行连贯。
- **⚡ 逐头计算 Attention**
  在 Prefill 阶段，以“单个 token 的单个 KV head”为基本计算单元，CPU 完成一个 head 的计算后再切换到下一个 head。该设计更契合 CPU 核数有限但缓存容量较大的硬件特性：尽可能让单个 head 的 KV 数据长期驻留在片上 Cache 中，从而减少重复内存加载。

## 🤖 支持模型
- ✅ Qwen3 系列
- ⏳ MiniMax M2.7

## Benchmark
我们已经与SGLang (CPU end) 的输出完全对齐，对比了性能


长程




## 📄 论文
如果你对 eLLM 的底层设计与技术细节感兴趣，欢迎阅读我们的[论文](ellm.pdf)并引用。需要说明的是，当前公开版本为**早期论文**，其中部分实现细节尚未完全反映 eLLM 的最新进展，我们正在持续更新中，敬请理解。


```bibtex
@misc{huangfu2025ellm,
  title        = {eLLM: Achieving Lossless Million-Token LLM Inference on CPUs Faster Than GPUs},
  author       = {Huangfu, Yaguang},
  howpublished = {Preprint, ResearchGate},
  year         = {2025},
  url          = {https://www.researchgate.net/publication/393416965}
}
```

## 📜 开源协议
这个项目使用 [Apache 2.0 License](LICENSE)。
