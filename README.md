# eLLM: Running Efficient Million-Token LLM Inference on CPUs
eLLM can run full-scale MoE models (Qwen3-480B) on a single-socket CPU-only server. It
- Runs real-time short-sequence inferences (100ms/token)
- Runs lossless long-sequence inferences up to **millions of tokens**

How It Works:
- Leverages the large memory capacity of CPUs to achieve extreme inference performance through a **space-for-time** design strategy

🌐 Language: [English](README.md) | [简体中文](README.zh-CN.md)

## ✅ Call for Contributors
* The project is under active development, with the **minimum prototype (Qwen30B)** expected to be released in about **1 month**!  
* We are currently looking for volunteers — if you're interested, please contact **lucienhuangfu@outlook.com**.

**Key Capabilities**:
* Full MoE model loading with dynamic expert activation
* Full storage for million-token context (KV Cache)
* Standard attention inference (every token can reference the full context)

## Use Cases
### Online Short-Text Inference
* Search-based Q&A
* Code completion
* Chatbots

### Offline Long-Text Inference (Deep Research)
- Code auditing / high-risk vulnerability detection
- Contract review / document compliance checking  
- Financial statement compliance checks
- Literary creation / extended writing

## Competitive Advantage: Private LLM Deployment Without GPUs

**eLLM enables small and medium teams to deploy large models with lower costs and more flexible setups.**

### No Need for High-End GPU Servers
- Single-socket CPU-only servers can run MoE-architecture LLMs  
- Requires only a general-purpose CPU supporting AVX512-F16  
- Memory can be expanded with DDR5

### Simple Deployment, Adaptable to Multiple Scenarios
- Easily deploy to local servers / private cloud / edge nodes  
- Supports on-demand elastic computation, automatically freeing resources after tasks  
- Scales horizontally to handle high-concurrency inference workloads

Machine Comparison: CPU-Only Server vs GPU Server

| CPU-Only Server | Item | GPU Server | |
|----------|--------------|------------|------|
|CPU ||CPU|GPU| 
| Xeon 6900| Model           | Xeon 8480+     | H20   |
|3| Memory Capacity (TB) | 2 | 0.141 |
|1| Quantity          |4        | 8  |
|15| Total Price (10k RMB) |150| 

## Problems with Existing Solutions
- High barrier, high cost
  - 🧠 **High GPU inference threshold**: per-user cost for long-context inference is extremely high
  - 📦 **Context limitations**: GPU VRAM cannot hold the full long-text context
  - 🔀 **Complex expert routing**: requires synchronized expert routing, adding system complexity
- Dynamic memory management and task generation consume additional CPU resources
  - CPU-only servers use a small portion of CPU for this, leaving most CPU cycles for computation
  - GPU servers require host CPU usage, but GPUs can focus solely on computation
- Performance bottlenecks grow **super-linearly** with sequence length
  - Dynamic memory allocation: pre-consumes memory, uncontrollable
  - Dynamic graph construction: high runtime overhead, low efficiency
  - Chunked KV cache: poor bandwidth utilization, lowering inference efficiency

## 💡 Why MoE is More Suitable for CPU-Only Inference

- MoE is a “store big, compute small” architecture
  - High storage demand: TB-level expert parameters must reside in memory  
  - Low bandwidth demand: only activated experts are loaded 
  - Low compute demand: only active paths are computed  
- CPU-only architecture matches MoE inference needs
  - **Large storage capacity**: easily holds all expert parameters  
  - **Low bandwidth capacity**: few activated experts, low memory bandwidth pressure
    - **MRDIMM**: doubles memory bandwidth, fully saturates compute pipelines  
  - **Low computation capacity**: ideal for AMX matrix acceleration
    - **AMX**: matrix instruction extensions, delivering multiple times performance improvement

## Why eLLM Outperforms Existing Frameworks (vLLM)

### Space-for-Time: Rebuilding CPU Inference Engines

#### 🛠️ Static Resource Allocation
- Static memory allocation + static graph compilation → static task set
- Directly obtain ready-to-run tasks, freeing more CPU cycles for computation

#### ⚡ Inference Latency Grows **Linearly** with Sequence Length
- Static memory allocation: avoids fragmentation, achieves globally optimal layout  
- Contiguous memory layout: maximizes KV Cache bandwidth utilization  
- Dynamic task scheduling: supports irregular input  
- Dynamic expert activation: only computes necessary paths, saving compute

## Roadmap
* [ ] Qwen (30B, 480B) (coming soon)
* [x] LLaMA 2 / 3
* [ ] DeepSeek (coming soon)
* [ ] gpt-oss

## 📄 Paper

If you're interested in the technical details, you can read our paper and cite:

```bibtex
@article{ellm2025,
  title={eLLM: Achieving Lossless Million-Token LLM Inference on CPUs Faster Than GPUs},
  author={Yaguang Huangfu},
  journal={preprint https://www.researchgate.net/publication/393416965},
  year={2025}
}
```

## 📜 License
This project is licensed under the [Apache 2.0 License](LICENSE).




