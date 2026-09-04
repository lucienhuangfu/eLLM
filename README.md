# eLLM: Run Long-Horizon Inference Faster on CPUs Than on GPUs
eLLM is an LLM inference framework for CPU servers. It adopts a "trade storage for computation" strategy, leveraging the CPU's large-capacity DDR memory to close the order-of-magnitude bandwidth gap against GPU HBM, and thereby delivers performance that surpasses GPUs in long-horizon inference.
- **Prefill**: achieves roughly **two orders of magnitude** of performance improvement over existing CPU inference frameworks
  - full single-pass Prefill for long text;
  - incremental Prefill on only the newly added input in multi-turn interactions;
- **Decode**: runs with a smaller batch, which not only activates fewer parameters but also gives each request a larger share of memory bandwidth, so inference speed can likewise exceed GPUs.

🌐 Languages: [English](README.md) | [简体中文](README.zh-CN.md)  
📚 Docs: [Documentation](docs/index.md)  
🎓 We currently have only 1–2 trainee openings and welcome applications from computer science students  
🛠️ The project is under active development; code is pushed to the main branch monthly  
💼 We are committed to open source and AI democratization and look forward to working with industry partners. Contact: **lucienhuangfu@outlook.com**

## 🚀 Progress and Updates
- `v0.0.3` (2026-09-04): Beta release; core features are complete, and inference results are fully aligned with SGLang CPU
- `v0.0.2` (2026-04-06): Alpha release
- `v0.0.1` (2025-12-20): Project open-sourced

## 🔑 Features
**eLLM**: an LLM inference framework for CPU servers
- Pure CPU inference, no GPU / NPU required
  - CPU: Intel Xeon / AMD EPYC (Xeon Gen4+ recommended)
  - Memory: ample DDR (sized to the model)
- vLLM API compatible, plugs directly into the existing ecosystem
- Inference results stay consistent with GPUs

## ✨ Advantages
eLLM outperforms GPU inference across a range of key metrics:
- **Low latency**: full Prefill and incremental Prefill significantly reduce time to first token (TTFT)
- **High throughput**: although single-instance concurrency is lower than GPU solutions, end-to-end latency is smaller, so **real QPS is actually higher**
- **Long context**: TB-scale large memory supports million-token, near-unbounded context windows
- **Low energy use**: Prefill loads parameters only once, greatly reducing the energy cost of repeated memory access
- **Low cost**: no water cooling or high-power supply needed; reuses existing CPU machines and data centers directly

## 🎯 Use Cases
eLLM is a strong fit for **long-horizon tasks**—Agent workflows that must keep goals, state, and reasoning consistent over long, multi-step execution:
- **Open Claw (Computer-use Agent)**
  - Dynamically loads skills and context on demand during execution, continuously planning, calling tools, and completing complex tasks
  - An online interactive Agent scenario emphasizing low latency, high-frequency tool calls, and long-term maintenance of task goals and execution state
- **Code Copilot**
  - Targets large cross-file, cross-module code repositories, supporting long-running software engineering tasks
  - Maintains project state across many rounds of editing, testing, debugging, and fixing—suited to Agent Coding scenarios such as refactoring, bug fixing, and code review
- **RAG (Retrieval-Augmented Generation)**
  - Dynamically retrieves and injects external knowledge during task execution, rather than loading all context up front
  - Ideal for enterprise knowledge-base QA, long-document analysis, and Agent workflows that require continuous retrieval and reasoning
- **Deep Research**
  - Supports multi-round retrieval, information synthesis, planning, and reasoning, continuously accumulating and reusing intermediate results over long research cycles
  - Suited to research tasks lasting hours or even days, not just single-shot long-context inference

## ⚙️ Approach
To better support **long-horizon tasks**, eLLM targets the Agent scenario of "multi-round execution + long-term state maintenance + low-latency interaction." Based on the CPU architecture profile (large memory, relatively weak compute), it proposes an overall "trade storage for computation" design philosophy. It restructures inference into a **reusable, continuously growing, locally incrementally updatable execution pipeline**, reducing repeated computation and state-rebuilding overhead in long tasks.

- 🧩 **Elastic static computation graph**
  Build a globally unique static computation graph and access tensors with a **dimension-first** layout, so that elements at the same logical coordinates map stably to the same memory location. This lets the same execution graph support different input lengths without rebuilding the graph.
- 🧊 **Static-shape KV cache (non-paged)**
  Preallocate a fixed-shape tensor for the KV cache instead of relying on paged block management. Locate KV directly by tensor coordinates when reading and writing, and read KV contiguously along the sequence dimension, reducing metadata maintenance, address mapping, and dynamic allocation overhead while avoiding TLB and cache misses as much as possible.
- 📦 **Massive-dimensional tensors**
  Reserve a large enough sequence dimension for tensors to build an effectively "unbounded" KV tensor, supporting full Prefill and thereby avoiding repeated Prefill and repeated parameter loading—suited to ultra-long prompts and long contexts.
- 🔁 **Session Cache**
  Keep KV state across multi-turn interactions and Prefill only the new input incrementally, with no need to recompute historical context. This provides "state continuity" at the mechanism level, supporting Agents in keeping context consistent and execution coherent over long tasks.


## 🤖 Supported Models
- ✅ Qwen3 series
- ⏳ Qwen3.8 (in development)

## 🚀 Quick Install and Chat

**Hardware requirements**
- CPU: AVX-512 FP16 support
- Memory: 128 GB+

**Software requirements**
- OS: Linux (x86-64)
- Rust: install with rustup; `rust-toolchain.toml` already pins nightly
- Python 3 and curl: used to run the chat client

We recommend first downloading the complete Qwen3-Coder-30B-A3B-Instruct model from Hugging Face.
After cloning the repository and entering its root directory, copy the model to the following path:
```text
models/Qwen3-Coder-30B-A3B-Instruct
```
Then build eLLM and start the service with roughly 50K token capacity and a single request slot:

```bash
git clone https://github.com/lucienhuangfu/eLLM.git
cd eLLM
# Copy the downloaded model to models/Qwen3-Coder-30B-A3B-Instruct
cargo build --release --bin main
./target/release/main \
  --model-path models/Qwen3-Coder-30B-A3B-Instruct \
  --chunk-size 50000 \
  --sequence-length 50000 \
  --batch-size 1
```

The first startup may take longer while the model weights and computation graph are initialized.
After loading weights and initializing the computation graph, the service listens on `0.0.0.0:8000`. In another terminal, run the streaming chat client:

```bash
python3 scripts/chat.py
```

Type `exit` or `quit` to end the conversation.

## 📊 Benchmark
The system has only been preliminarily optimized so far and still has a lot of headroom. But experiments show that in short-horizon tasks (single-turn interaction), eLLM already leads the CPU baseline (SGLang CPU backend) across the board, and the advantage keeps widening as input length grows:
- **Prefill (TTFT, s)**: executed continuously in one pass with no segment jumps, **12%–73% faster** than the chunked CPU baseline, and also ahead of the unchunked baseline overall (max gap about 12%)
- **Decode (TPOT, s/token)**: a steady **1.5×–1.6× speedup** over the CPU baseline, with the lowest growth slope throughout

As context keeps growing, Prefill and Decode are expected to surpass GPUs; long-horizon task (multi-turn interaction) experiments are underway.
For detailed experimental setup and data, see [benchmark.md](benchmark.md).

## 📄 Paper
If you are interested in the underlying design and technical details of eLLM, please read and cite our [paper](ellm.pdf). Note that the current public version is an **early paper**, and some implementation details do not yet fully reflect the latest progress of eLLM. We are continuing to update it—thank you for your understanding.

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
This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).
