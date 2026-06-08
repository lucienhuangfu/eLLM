---
hide:
  - navigation
  - toc
---

# Welcome to eLLM

eLLM is a high-performance, CPU-focused LLM inference engine written in Rust.
It provides an [OpenAI-compatible](https://platform.openai.com/docs/api-reference/chat) HTTP serving layer
on top of a hand-tuned multi-threaded inference runtime — no GPU required.

---

## Where to start

- **Run a model right now** → [Quickstart](getting_started/quickstart.md)
- **Build from source** → [Installation](getting_started/installation.md)
- **Understand the architecture** → [Design Overview](design/index.md)
- **Add a new operator** → [Contributing](contributing/new_operator.md)

---

## Highlights

- Pure Rust inference runtime with AVX-512 / AVX2 SIMD kernels
- OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming
- Event-driven batch scheduler — threshold + timeout dual-trigger
- Supports dense and sparse-MoE transformer families (Qwen3, MiniMax-M2.5, Llama-2, Mixtral)
- Golden-file HuggingFace alignment workflow for operator validation
