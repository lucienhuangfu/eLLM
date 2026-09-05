# eLLM：让 CPU 在长程推理中快过 GPU
eLLM 是一款面向 CPU 服务器的大模型推理框架。它采用“以存换算”策略，利用 CPU 大容量 DDR 内存，弥补其与 GPU HBM 一个数量级的带宽差距，从而在长程任务推理场景下实现超越 GPU 的性能。
- **Prefill**：相较现有 CPU 推理框架可实现约**两个数量级**的性能提升
  - 长文本一次性整段 Prefill——不分块、不重复加载参数；
  - 多轮交互中保留上下文 KV，仅对新增输入做增量 Prefill——不重复计算历史轮次；
- **Decode**：以更小的 batch 运行，不仅激活的参数更少，单个 request 可分得的内存带宽也更高，因此推理速度同样可以超过 GPU。

🌐 语言版本：[English](README.md) | [简体中文](README.zh-CN.md)  
🎓 目前仅开放 1–2 个 Trainee 名额，欢迎计算机专业在校生报名  
🛠️ 项目正在紧张开发中，代码会按月推送到 main 分支  
💼 我们致力于推动开源与 AI 民主化，期待与产业携手合作，联系方式：**lucienhuangfu@outlook.com**

## 🚀 进展与更新
- `v0.0.3`（2026-09-04）：发布 Beta 版本，核心功能开发完成，推理结果与 SGLang CPU 完全对齐
- `v0.0.2`（2026-04-06）：发布 Alpha 版本
- `v0.0.1`（2025-12-20）：项目开源

## 🔑 功能
**eLLM**：面向 CPU 服务器的大模型推理框架
- 纯 CPU 推理，无需 GPU / NPU
  - CPU：Intel Xeon / AMD EPYC（推荐 Xeon Gen4+）
  - 内存：足量 DDR（按模型规模配置）
- 兼容 vLLM API，可直接接入现有生态
- 推理结果与 GPU 保持一致

## ✨ 优势
eLLM 在多项关键指标上全面超越 GPU 推理：
- **低延迟**：通过整段 Prefill 与增量 Prefill，显著降低首 token 延迟（TTFT）
- **高吞吐**：单实例并发度虽低于 GPU 方案，但端到端延迟更小，**实际 QPS 反而更高**
- **长上下文**：TB 级大内存支撑百万 token 级、近乎无限长度的上下文窗口
- **低能耗**：Prefill 阶段参数仅需加载一次，大幅减少重复访存带来的能耗
- **低成本**：无需水冷散热与大功率供电，直接复用现有 CPU 机器与机房

## 🎯 应用场景
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
为了更好地支持**长程任务（Long-Horizon Tasks）**，eLLM 针对“多轮执行 + 长时间状态维护 + 低延迟交互”的 Agent 场景，基于 CPU 体系结构（内存大、算力小）提出“以存换算”的整体设计理念。它将推理过程重构为**可复用、可持续增长、可局部增量更新的执行链路**，从而降低长任务中的重复计算与状态重建开销。

- 🧩 **弹性静态计算图**
  构建全局唯一的静态计算图，并采用 **维度优先（dimension-first）** 的布局存取张量，让相同逻辑坐标的元素稳定映射到相同内存位置，使同一套执行图可以在不重建计算图的前提下支持不同输入长度。
- 🧊 **静态形状 KV Cache（不分页）**
  为 KV Cache 预分配固定形状的 tensor，不依赖分页式 block 管理；读写时直接按张量坐标定位 KV，并沿 sequence 维度连续读取 KV，减少元数据维护、地址映射和动态分配开销，尽量避免 TLB miss 和 cache miss。
- 📦 **超大维度张量**
  为张量预留足够大的 sequence 维度，构建近似“无限长度”的 KV tensor，支持整段 Prefill，从而尽量避免重复 Prefill 和参数反复载入，适配超长 Prompt 和长上下文。
- 🔁 **Session Cache**
  在多轮交互中持续保留 KV 状态，仅对新输入进行增量 Prefill，而无需重复计算历史上下文；从机制上实现“状态连续性”，支撑 Agent 在长时间任务中保持上下文一致与执行连贯。

## 🤖 支持模型
- ✅ Qwen3 系列
- ⏳ Qwen3.8（开发中）

## 🚀 快速安装与对话

**硬件要求**
- CPU：支持 AVX-512 FP16
- 内存：128 GB+

**软件要求**
- 操作系统：Linux（x86-64）
- Rust：用 rustup 安装即可，`rust-toolchain.toml` 已指定 nightly
- Python 3 与 curl：用于运行对话客户端


建议先从 Hugging Face 下载完整的 Qwen3-Coder-30B-A3B-Instruct 模型。
克隆仓库并进入根目录后，将模型复制到以下路径：
```text
models/Qwen3-Coder-30B-A3B-Instruct
```
随后编译 eLLM，并以约 50K token 容量和单请求槽位启动服务：

```bash
git clone https://github.com/lucienhuangfu/eLLM.git
cd eLLM
# 将下载好的模型复制到 models/Qwen3-Coder-30B-A3B-Instruct
cargo build --release --bin main
./target/release/main \
  --model-path models/Qwen3-Coder-30B-A3B-Instruct \
  --chunk-size 50000 \
  --sequence-length 50000 \
  --batch-size 1
```

首次启动需要初始化模型权重和计算图，耗时可能较长。
服务完成权重加载和计算图初始化后监听 `0.0.0.0:8000`。在另一个终端运行流式对话客户端：

```bash
python3 scripts/chat.py
```

输入 `exit` 或 `quit` 结束对话。


## 📊 Benchmark
系统目前仅完成初步优化，仍有很大提升空间。但实验表明，在短程任务（单轮交互）中，eLLM 对 CPU baseline（SGLang CPU backend）已全面领先，且优势随输入长度增加持续扩大：
- **Prefill（TTFT, s）**：整段连续执行、无分段跳变，相比 chunked CPU baseline 快 **12%～73%**，对 unchunked baseline 亦整体占优（最大差距约 12%）
- **Decode（TPOT, s/token）**：相比 CPU baseline 稳定提速约 **1.5×～1.6×**，且增长斜率全程最低

随着上下文继续增长，Prefill 与 Decode 有望快过 GPU；长程任务（多轮交互）实验进行中。
详细实验设置与数据见：[benchmark.zh-CN.md](benchmark.zh-CN.md)。

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
本项目基于 [GNU Affero General Public License v3.0](LICENSE)（AGPL-3.0）开源。
