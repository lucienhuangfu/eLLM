eLLM 让纯 CPU 服务器可实现比 GPU 更快的 LLM 推理。在以 Prefilling 为主的长文本场景中，单颗 CPU（Xeon）可快过 8 块 GPU（H20）。
- 依托 CPU 的大内存，Prefill 整段长文本，避免分段处理和重复载入参数。
- 依托 CPU 的大 Cache，逐头计算 Attention，减少重复载入 KV;
eLLM 让 CPU（Xeon / EPYC）成为最佳的 AI 推理芯片。
GitHub: https://github.com/lucienhuangfu/eLLM

eLLM enables CPU-only servers to deliver LLM inference faster than GPUs. In long-context inference workloads dominated by Prefilling, a single CPU (Xeon) can outperform 8 GPUs (H20).
- With their large memory capacity, eLLM prefills the entire long prompt in a single pass, avoiding chunked execution and repeated parameter loading;
- With their large cache, eLLM computes attention head by head, reducing repeated KV loads.
eLLM makes CPUs （Xeon / EPYC） the best chips for AI inference.
GitHub: https://github.com/lucienhuangfu/eLLM
