# Chinese
eLLM 让纯 CPU 服务器可实现比 GPU 更快的 LLM 推理。在以 Prefilling 为主的长文本场景中，单颗 CPU（Xeon）可快过 8 块 GPU（H20）。
- 依托 CPU 的大内存，Prefill 整段长文本，避免分段处理和重复载入参数;
- 依托 CPU 的大 Cache，逐头计算 Attention，减少重复载入 KV。
eLLM 让 CPU（Xeon / EPYC）成为最佳的 AI 推理芯片。
GitHub: https://github.com/lucienhuangfu/eLLM









# v1
eLLM can run LLM inference on CPUs faster than on GPUs. 
In long-context workloads dominated by Prefilling, a single CPU server (Xeon) can outperform an 8-GPU H20 server.
GitHub: https://github.com/lucienhuangfu/eLLM


# v2
- With its large memory capacity, eLLM can prefill the entire long prompt in a single pass, avoiding chunked execution and repeated parameter loading;
- With its large cache, eLLM computes attention head by head, reducing repeated KV loads.

eLLM makes CPUs (Xeon/EPYC) the best AI inference chips.


eLLM can infer LLM on CPUs faster than on GPUs
