# eLLM

## eLLM: Making LLM Inference on CPUs Faster Than GPUs

### eLLM — Turn Xeon/EPYC into optimal AI inference chips

Mission: Break the GPU barrier so powerful AI is accessible to everyone.

👉 Project: https://github.com/lucienhuangfu

🌐 Language versions: English (this file) | 简体中文 ([README.zh-CN.md](README.zh-CN.md))

We are seeking volunteers and funding support.
📧 Contact: lucienhuangfu@outlook.com

## 🚀 Progress & Releases
- [v0.1.0-alpha.1] (2026-04-06) Alpha release
- [v0.0.1] (2025-12-20) Open sourced

## 🔑 Key Features
eLLM is a large-model inference framework optimized for CPU-only servers:
- Pure CPU inference — runs on CPU servers (Xeon / EPYC) with no GPU/NPU required
- vLLM-compatible API — integrates smoothly with existing ecosystems
- Numerically and behaviorally equivalent to GPU inference where applicable

## Hardware (no GPU/NPU required)
- CPU: Intel Xeon Gen4 or newer (AMX support recommended)
- Memory: ample DDR5 (no HBM required)

## ✨ Advantages
eLLM leverages CPU architecture strengths for inference and achieves improvements over GPU-based inference across multiple dimensions:
- Low latency: supports full-sequence Prefill to reduce time-to-first-token
- High throughput: lower end-to-end latency can yield higher effective QPS
- Very long context: large memory enables near-“unbounded” context windows
- Lower energy: Prefill loads parameters once, reducing repeated memory I/O
- Lower cost: reduced hardware and per-inference costs versus GPU solutions

## Target Applications
eLLM’s strengths (long context, long-lived sessions, low latency) make it especially well suited for:
- Code Copilot: cross-file long-context code understanding and long-lived edit sessions
- RAG (Retrieval-Augmented Generation): large external documents retained in context to avoid repeated Prefill
- Deep Research: multi-step retrieval and reasoning with long-lived intermediate state
- Deep Thinking: chain-of-thought or tree-of-thought workloads with large amounts of intermediate state

## ⚙️ Methodology
eLLM embraces a "memory-for-compute" design for CPU servers (large memory, relatively lower compute bandwidth). It restructures inference into a preallocated, directly addressable, and reusable execution pipeline, trading modest runtime compute for much lower runtime overhead.

- 🧩 Elastic static computation graph: a single global static compute graph with dimension-first tensor layout that ensures consistent memory addresses for identical logical coordinates. The graph can handle varying input lengths without rebuilding.
- Static-shaped KV cache (no paging): KV cache tensors are preallocated with fixed shapes and accessed by direct tensor coordinates, avoiding page/block metadata, address remapping, and dynamic allocation.
- 📦 Very large-dimension tensors: reserve large token/sequence dimensions so Prefill can be performed as a single continuous operation, minimizing repeated Prefill and parameter reloads.

## 🤖 Supported Models
- Qwen3 series
- MiniMax M2.5

## Experiments
The minimal prototype has been implemented. We designed short- and long-text experiments comparing Prefill and Decode phases between a single CPU server and an 8-GPU inference node. Short-text decode shows clear CPU improvements over existing CPU baselines; long-text Prefill shows eLLM can outperform multi-GPU systems thanks to large-memory advantages.

### Notes on current status
- Operator-level tests and alignment are complete.
- Model outputs are not yet fully matched to reference implementations.
- Current runs use randomly-initialized parameters and do not yet include attention and tokenization integration.

### Short-text experiments (completed)
- Model: Qwen3-Coder-30B-A3B-Instruct (FP16)
- Scenario: short prompts, batch=1, prompt lengths {128, 256, 512}
- Only Decode was benchmarked (Prefill not included). eLLM outperforms the SgLang CPU baseline on TPOT (time per output token), achieving ~1.6× speedup (~38% latency reduction).

Analysis: Short-text decode bottlenecks are dominated by scheduling, memory management, and runtime control-path overheads rather than raw operator FLOPs. eLLM’s static compute graph and lean execution path reduce dynamic scheduling and state maintenance costs.

Key CPU baseline overheads:
- Scheduling overhead (continuous batching, token-level routing, request merge/split)
- KV cache management (block allocation, address mapping)
- Intermediate tensor allocation and fragmentation
- Service/runtime overheads (API, streaming, context switching)

### Long-text experiments (ongoing)
Long-context workloads favor CPU large-memory strategies: GPUs must chunk long contexts and repeatedly reload parameters, increasing Prefill cost. eLLM aims to Prefill end-to-end as a single continuous pipeline, reducing repeated I/O and control/synchronization overhead.

Experiment setup (example):
- Model: Qwen3-Coder-480B-A35B-Instruct (FP16)
- Scenario: batch=10, prompt length=100,000 tokens
	- eLLM: chunk size = 1,000,000, batch=10, sequence length=100,000 (single-shot Prefill)
	- GPU baseline: chunk size = 10,000, batch=10, sequence length=1,000 (100 segments)

Prefill analysis summary:
- Parameter and KV loads: GPUs repeatedly load parameters across chunks, adding I/O overhead. CPU main memory often holds more data at once and reduces repeated loads.
- KV locality and organization: eLLM stores KV in fixed-shaped, dimension-first tensors and processes per-head sequentially, improving cache locality compared to multi-head parallel GPU strategies.
- Chunking cost: segmenting long contexts increases scheduling, synchronization, and intermediate state maintenance; continuous Prefill reduces these control-path costs.

Decode analysis summary:
- Although GPU raw bandwidth (HBM) is higher, small-batch long-context decode exposes GPU inefficiencies (reduced parallelism, uneven MoE load, kernel-launch overhead, fragmented KV access). eLLM reduces these inefficiencies on CPU due to larger memory, steady cache behavior, and fewer kernel-launch-style overheads.

## Conclusion
While GPUs have dominated LLM inference, eLLM shows that for long-context workloads, a single CPU server can match or exceed multi-GPU systems by exploiting large-memory Prefill, reducing repeated I/O, and minimizing control-path overhead. Extending eLLM to multi-socket NUMA servers may further improve performance for many inference workloads.

## 📄 Paper
If you are interested in the design details, please read and cite our preprint (early version). Note this public version is an early draft and may not reflect the latest implementation details; we are continuously updating it.

```bibtex
@misc{huangfu2025ellm,
  title        = {eLLM: Achieving Lossless Million-Token LLM Inference on CPUs Faster Than GPUs},
  author       = {Huangfu, Yaguang},
  howpublished = {Preprint, ResearchGate},
  year         = {2025},
  url          = {https://www.researchgate.net/publication/393416965}
}
```

## 📜 License
This project uses the [Apache 2.0 License](LICENSE).

## Operators taxonomy

`src/operators` now exposes 5 semantic categories to make operator responsibilities easier to understand and maintain:

- `operators::linear`: dense linear algebra (`MatMul`, `MatMul3`, `MatMulAdd`, `Attention`)
- `operators::routing`: token/expert routing and candidate selection (`MatMulTopK`, `ExpertsSoftmaxNorm`, `TopKSoftmax`)
- `operators::expert`: expert-network compute and merge (`ExpertsMatMulSilu`, `ExpertsMatMulDown`, `ExpertsMergeAdd`)
- `operators::transform`: elementwise + normalization transforms (`AddZipMap`, `SiluMulZipMap`, `ComplexZipMap`, `RMSMap`, `LookupRMSMap`, `AddRMSZipMap`)
- `operators::movement`: state/data movement operators (`LiftVector`)
- `operators::traits`: centralized trait definitions used by all operator categories

### Classification criteria

- **Primary math semantics first**: grouped by what the operator computes, not by where SIMD/scalar kernels live.
- **Pipeline role second**: routing, expert execution, transform, and movement are separated by inference-stage responsibility.
- **Stable call-site ergonomics**: runtime code can import from semantic buckets without needing knowledge of internal file layout.
- **Trait centralization**: all trait definitions are merged under `src/operators/traits` for unified maintenance.

## Docs

Selected design notes live under [`docs/`](docs/):

- [`docs/matmul.md`](docs/matmul.md): high-performance matrix multiplication principles and how they map to this repository
- [`docs/attention.md`](docs/attention.md): attention scheduling and static parallel decomposition
- [`docs/minimal_model_abstraction.md`](docs/minimal_model_abstraction.md): minimal model abstraction strategy


