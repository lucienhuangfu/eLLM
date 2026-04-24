# eLLM：让 LLM 推理在 CPUs 上快过 GPUs
## eLLM：让 CPU（Xeon/EPYC）成为 AI 推理芯片的首选
👉 项目主页：[https://github.com/lucienhuangfu/eLLM](https://github.com/lucienhuangfu/eLLM)  
🌐 语言版本：[English](README.md) | [简体中文](README.zh-CN.md)  
🎓 目前仅开放 1–2 个 Trainee 名额，欢迎计算机专业在校生报名  
💼 我们致力于推动开源与 AI 民主化，期待与产业携手合作  
📧 联系方式：**lucienhuangfu@outlook.com**

## 🚀 进展与更新
- `v0.1.0-alpha.1`（2026-04-06）：发布 Alpha 版本
- `v0.0.1`（2025-12-20）：项目开源

## 🔑 功能
**eLLM**：专为 **CPU 服务器**打造的大模型推理框架
- **纯 CPU 推理**：运行在 **CPU 服务器**（Xeon/EPYC）上，**无需 GPU/NPU**
- **兼容 vLLM API**：可无缝接入现有生态
- **与 GPU 一致的推理**：在数值结果和运行行为上与 GPU 推理一致

## 硬件要求（无需 GPU/NPU）
- **CPU**：Intel Xeon Gen4 及以上（支持 AMX 指令集）
- **内存**：足量的 DDR5（无需 HBM）

## ✨ 优势
eLLM 充分释放了 **CPU 在推理场景下的体系结构优势**，使其在多项关键指标上实现对 GPU 推理的全面超越：
- **低延迟**：整段 Prefill，逐 head 计算 attention，显著降低首 token 延迟
- **高吞吐**：单实例并发度虽低，但由于端到端延迟更小，**实际 QPS 反而更高**
- **长上下文**：大内存支持近乎无限长度的上下文窗口
- **低能耗**：Prefill 阶段仅加载一次参数，大幅降低重复访存的能耗
- **低成本**：硬件成本与单用户推理成本显著低于 GPU 方案

## 应用场景
eLLM 适合以 **Prefill** 为主的长文本推理任务，尤其适合需要超长上下文的 Agent：
- **Code Copilot**
  - 面向跨文件、跨模块的超长代码上下文，可一次性完成 Prefill
  - 适合重构、代码审查、补全等高频增量推理场景
- **RAG（Retrieval-Augmented Generation）**
  - 可将大规模外部文档与知识库动态注入长上下文
  - 适合超长文档理解与企业知识库问答
- **Deep Research**
  - 适合多步骤检索、推理与信息整合，首轮即可装入大量资料
  - 支持跨数小时甚至数天的连续研究流程

## ⚙️ 方法
基于 CPU 服务器“内存大、缓存大、计算小”的体系结构，eLLM 采用“**以存换算**”的设计理念，重构大模型推理框架，将推理过程压缩为一条可预分配、可直接访问、可稳定复用的执行链路，以更低的运行时开销换取更稳定的端到端延迟。

- 🧩 **弹性静态计算图**
  构建全局唯一的静态计算图，并采用 **维度优先（dimension-first）** 的布局存取张量，让相同逻辑坐标的元素稳定映射到相同内存位置，使同一套执行图可以在不重建计算图的前提下支持不同输入长度。
- **静态形状 KV Cache（不分页）**
  为 KV Cache 预分配固定形状的 tensor，不依赖分页式 block 管理；读写时直接按张量坐标定位 KV，并沿 sequence 维度连续读取 KV，减少元数据维护、地址映射和动态分配开销，尽量避免 TLB miss 和 cache miss。
- 📦 **超大维度张量**
  为张量预留足够大的 token/sequence 维度，构建近似“无限长度”的 KV tensor，支持整段 Prefill，从而尽量避免重复 Prefill 和参数反复载入，适配超长 Prompt 和长生命周期上下文。
- **⚡ 逐头计算 Attention（FlashAttention）**
  在 Prefill 阶段，以“单个 token 的单个 KV head”为基本计算单元，CPU 完成一个 head 的计算后再切换到下一个 head。该设计更契合 CPU 核数有限但缓存容量较大的硬件特性：尽可能让单个 head 的 KV 数据长期驻留在片上 Cache 中，从而减少重复内存加载。


## 🤖 支持模型
- ✅ Qwen3 系列
- ✅ MiniMax M2.5

## 实验
截至目前，eLLM 的最小原型已经完成。为验证它的性能潜力，我们设计了短文本与长文本两类实验，并分别考察 Prefill 和 Decode 两个阶段，比较单块 CPU 服务器与由 8 块 GPU 组成的推理节点在不同场景下的表现。短文本推理场景下，CPU 明显落后于 GPU；但在长文本推理场景下，eLLM 有机会凭借 CPU 的大内存优势实现反超。

### 实验设置
- CPU baseline：SGLang CPU endpoint（单块 CPU 服务器）
- GPU baseline：SGLang GPU endpoint v0.5.9（多卡 GPU 服务器，示例使用 8x H20 节点）
- 当前实验运行在公有云上的 CPU 虚拟机中，仅占服务器的一小部分资源
- Prefill 指标：TTFT（Time to First Token，ms）
- Decode 指标：TPOT（Time Per Output Token，ms/token）

### 硬件对比

| 条目 | CPU 服务器 | GPU 服务器 |
|---|---|---|
| 型号 | Xeon 6982P-C | H20 |
| 核数 | 128 | 16,000 |
| FP16 矩阵算力（TFLOPS） | 250 | 296 |
| Cache（MB） | 504（L3） | 60（L2） |
| 最大内存容量（TB） | 3 | 0.141 |
| 数量 | 1 | 8 |
| 总价（$） | 17,000 | 220,000 |



### 实验说明
- 当前实验聚焦于 **benchmark 与系统性能评估**。
- **算子层面**已完成测试与对齐，说明底层执行链路已经具备基本可用性。
- **模型层面**的输出未与参考实现完全一致。
  - 当前加载的是 **随机初始化参数**，尚未接入真实模型权重。
  - 本阶段尚未纳入 **attention** 和 **切词（tokenization）** 流程。


#### 短文本实验（已完成）

**实验设置**
- 模型：Qwen3-Coder-30B-A3B-Instruct（FP16）
- 场景：短 Prompt，`batch = 1`，`prompt length = {128, 256, 512}`
- 由于短文本场景下，所有 CPU 推理框架在 Decode 性能上通常都明显落后于 GPU，因此本组实验不再单独加入 GPU 对比。
- 本组只做 Decode 实验，不做 Prefill；Prefill 的效果放到长文本实验中验证。

**实验环境**

| 条目 | 配置 |
|---|---|
| 环境 | CPU 虚拟机 |
| 配置核数 | 48 |
| 配置内存容量（TB） | 0.192 |

**Decode 结果**
在 `prompt_len=128/256/512` 的三组测试中，eLLM 均稳定优于 SgLang CPU baseline，在 CPU 上表现出更低的 TPOT。综合来看，eLLM 约带来 `1.6×` 的性能提升，对应约 `38%` 的延迟下降。随着上下文长度增加，两者的 TPOT 都呈近似线性增长，但 eLLM 的斜率更低，说明其在短上下文范围内已经展现出更好的效率趋势。

```mermaid
xychart-beta
    title "TPOT Comparison (ms/token, Lower is Better)"
    x-axis [128, 256, 512]
    y-axis 0 --> 60
    line "eLLM (CPU end)" [32.94, 33.01, 33.13]
    line "SgLang (CPU end)" [52.5, 52.47, 52.71]
```

**Decode 分析**  
这一结果表明，短文本 decode 的瓶颈并不主要落在算子计算本身，而更多来自调度、内存管理和运行时这些“控制路径”开销。eLLM 的静态计算图和更轻量的执行路径减少了动态调度与状态维护成本，把更多时间留给真正的算子执行，因此能够在 CPU baseline 上获得稳定收益。

从 CPU baseline 的执行链路看，主要损耗可以归纳为四类：

- 调度开销：需要频繁执行 continuous batching、token 级路由以及请求合并/拆分；每生成一个 token 都要经过一次调度路径，随着活跃请求增多，控制开销会持续上升。
- KV Cache 管理：自回归 Decode 需要持续保存历史 token 的 KV 状态，并处理 KV block 的分配、回收和地址映射；这些操作单次开销不大，但频率极高，容易放大元数据和访存成本。
- 中间张量管理：Decode 过程中仍会产生 Q、K、V 投影、attention 中间结果、MLP 激活和 residual buffer 等临时 tensor；如果不能稳定复用，就会引入频繁分配与释放、内存碎片和带宽压力。
- 服务框架/运行时开销：API 服务、请求生命周期和 streaming 调度都会带来额外成本；GIL、上下文切换和动态数据结构操作也会进一步拖慢端到端延迟。

#### 长文本实验（预计 5 月底完成）
GPU 显存容量较小，chunk size 受限，使得长 Prompt 必须分段处理，同时也限制了 batch size 的规模。在 Prefill 阶段，需要对分段后的长上下文进行重复处理，带来额外开销。在 Decode 阶段，小 batch size 会导致并行度不足，从而引起性能明显下降。


**实验设置**
- 模型：Qwen3-Coder-480B-A35B-Instruct（FP16）
- 场景：`batch size = 10`，`prompt length = 100,000`
  - eLLM：`chunk size = 1,000,000`，`batch size = 10`，`sequence length = 100,000`，整段完成
  - GPU baseline：`chunk size = 10,000`，`batch size = 10`，`sequence length = 1,000`，需要分 100 段完成

**结果**  
目前实验数据仍在收集与整理中，尚未形成最终结论。

**Prefill 分析**  
eLLM 预计会显著快于 GPU baseline。在超长 Prompt 的 Prefill 阶段，首 token 延迟（TTFT）主要由两类因素驱动：其一是大规模的数据读取（模型参数与 KV 的加载），其二是分段处理带来的调度与同步开销。eLLM 的目标是将 Prefill 组织为尽可能连续、低干预的流水线，从根本上压缩这两类开销。若 eLLM 能稳定支持整段 Prefill，就有望将“连续访问、减少重复载入、降低控制开销”的优势转化为可观的首 token 延迟下降。下面按因果链逐项说明：

- **1) 参数与 KV 的读取：**
  - 问题：对于超长输入，若显存无法一次容纳，GPU 往往需要将输入拆成多个 chunk 顺序处理。受分段策略和显存管理限制，每个 chunk 的处理都可能重复将模型参数及相关 KV 加载到 GPU 缓存中，带来多次内存 I/O，从而累积出显著延迟。  
  - eLLM 优势：服务器级 CPU 通常拥有更大的主内存，能够用更少的分段甚至一次性完成 Prefill，显著降低重复内存 I/O。尽管 CPU 的 DDR5 带宽低于 GPU 的 HBM，但通过减少重复载入，TTFT 通常能获得更明显的改善。

- **2) KV 组织与访问模式：**
  - 问题：在超长上下文场景下，KV cache 规模随序列长度线性增长，其访问模式（按 head 或按 token）直接决定缓存命中率与内存搬运开销。在 GPU many-core 架构下，为最大化吞吐能力，通常采用 batch × head 维度的高度并行计算方式，并在计算过程中需要同时驻留多个 KV head 数据。这会显著增加 cache footprint，导致更频繁的 cache eviction，并加剧 HBM 带宽竞争与数据搬运开销，尤其在超长上下文推理场景中问题会被进一步放大。  
  - eLLM 优势：CPU 具备显著更大的片上缓存（L3 cache），在 KV 数据驻留能力与访问局部性方面具有天然优势。基于这一特性，eLLM 采用固定形状、维度优先的 KV 存储布局，并在执行策略上采用“逐 head 顺序计算”。在 CPU 实现中，各计算核心先完成某一 attention head 的全部 token 计算，期间该 head 的 KV 数据能够长期驻留于 cache 中并被复用，随后再切换至下一个 head。该设计使单个 head 的 KV 在 cache 中获得更长的驻留时间窗口与更连续的访问路径，从而显著增强时间局部性与空间局部性。得益于 CPU cache 的容量特性与 eLLM 的访问模式优化，单个 KV head 的有效 cache 驻留能力相较常规并行执行模式可提升约 **2–3 个数量级**。

- **3) 分段带来的控制与同步成本：**
  - 问题：将长 Prompt 切成多个 chunk 会引入额外的调度点、同步开销、内存碎片和跨段中间态维护（例如 KV 重组与合并），这些都会直接增加首 token 的延迟。  
  - eLLM 优势：若能把 Prefill 做成一次连续的流水（整段 Prefill），就可以显著减少调度与同步点，从而把控制路径开销降到最低。  

**Decode 分析**  
在长上下文的 decode 阶段，eLLM 虽然整体性能仍低于 GPU baseline，但两者之间的差距显著小于 DDR 与 HBM 的理论带宽差距。这表明，在该场景下，GPU 的带宽优势并未被充分发挥，其性能瓶颈更多来源于并行度不足与访存模式不理想，而非纯粹的带宽上限。

- **1) batch size 小：**
  - 问题：长序列会直接压缩 batch size。在 GPU 显存受限、chunk size 固定的情况下，sequence length 越长，可同时容纳的 batch size 越小。decode 阶段每个请求都需要携带完整历史 KV cache，使得有效并发进一步下降。
  - eLLM 优势：CPU 内存容量更大，支持更大的 chunk size，batch size 不易受限，从而能够维持更高的并发度。

- **2) 瘦矩阵乘法效率低：**
  - 问题：由于 batch size 变小，线性层的矩阵乘法会退化为瘦矩阵乘法；在 MoE 场景下，部分专家计算甚至会进一步退化为向量-矩阵乘法。这类算子通常计算规模小、形状不规则，GPU 难以充分发挥大规模并行优势；同时受限于 L2 cache 容量，也不容易同时维持足够多的并行矩阵乘法。
  - eLLM 优势：CPU 对小 batch、低维度和不规则形状的矩阵运算更友好，执行路径也更稳定。再加上更大的 L3 Cache，权重和中间结果更容易复用，因此可以更高效地并行处理多个矩阵乘法。

- **3) MoE 负载不均：**
  - 问题：MoE 专家分布在不同 GPU 上，而专家激活具有随机性。在小 batch 场景下，容易导致专家负载分布不均，部分 GPU 过载而其他 GPU 空闲，甚至退化为仅少数 GPU 在工作。
  - eLLM 优势：eLLM 在单机 CPU 上运行，无需跨设备分布专家，避免了负载不均和跨设备通信问题，能够稳定利用全部计算资源。

- **4) 有效内存带宽不理想：**
  - 问题：GPU 的高带宽依赖于大量 warp 并发执行以及持续的 memory-level parallelism 来隐藏内存访问延迟。在小 batch 场景下，可调度的 warp 数量不足，SM 无法被充分占满，内存请求流不连续，HBM 延迟暴露，SM 频繁因等待数据而停顿，从而导致有效带宽显著下降。
  - eLLM 优势：CPU 核数较少，对并行度要求更低，即使在小 batch 场景下也能较容易填满计算资源；同时配合 cache 与预取机制，可以更稳定地接近理论内存带宽。

- **5) 访存效率低：**
  - 问题：Paged KV Cache 进一步降低访存效率。KV 被拆分为离散 page 后，原本连续的访问被打散，破坏 memory coalescing，降低访存合并效率。同时需要通过 page table 进行地址映射，并伴随 pointer chasing，引入额外 load 和更长依赖链，降低指令级并行性。此外，还会带来 TLB miss 和 cache miss，非连续访问使得同样数据需要更多 memory transaction 才能完成加载，进一步放大带宽消耗。
  - eLLM 优势：采用静态连续 KV tensor，通过坐标直接访问，实现线性访存模式，能够充分利用硬件 prefetch 和 cache，提高整体访存效率。

- **6) Kernel launch/调度开销放大：**
  - 问题：decode 是逐 token 推进的过程，每一步都会触发一系列 GPU kernel（attention、matmul、layernorm 等）。在小 batch 场景下，单次 kernel 计算量较小，但 kernel launch 与调度开销不变，导致其占比显著上升。同时由于计算粒度过小，GPU 难以形成持续饱和的执行流水线，utilization 波动明显，SM 无法长期满载，整体吞吐下降。
  - eLLM 优势：eLLM 在 CPU 上以函数调用方式执行，无需 kernel 启动开销，在小 batch 和低并行度场景下具有更稳定的执行效率。


## 结论

GPU 长期以来被视为大模型推理的主流选择，而 CPU 往往被认为难以在同一赛道上竞争。eLLM 的实验结果表明，这一判断并不总是成立：在长文本推理场景中，单块 CPU 也有机会在端到端性能上与多卡 GPU 系统正面竞争，甚至实现反超。

其根本原因在于，eLLM 充分利用了 CPU 的两项核心硬件优势。第一，CPU 拥有更大的主内存，能够支持整段长 prompt 的 Prefill，减少分段处理带来的重复载入与调度开销。第二，CPU 具备更大的 Cache 空间，配合逐 head 计算 attention 的执行方式，可以显著提升数据驻留与复用效率，使 Prefill 阶段获得更低的整体延迟。

因此，在 Prefill 占主导的推理任务中，即便 Decode 阶段略慢，Prefill 的优势仍然足以主导总体耗时，最终带来更好的端到端表现。进一步看，如果将 eLLM 扩展到 NUMA 架构的多路 CPU 服务器上，并结合更大规模的内存与并行资源，它有望覆盖更多长上下文、长生命周期、低延迟的推理场景，形成一条区别于 GPU 路线的高性价比推理方案。


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
