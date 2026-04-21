eLLM 让纯 CPU 服务器也能跑出比 GPU 更快的 LLM 推理。  
在特定长文本场景下，单块 CPU （Xeon） 甚至可快过 8 块 GPU（H20）。
- "以存换算"重构纯 CPU 推理框架
- 依托 CPU 大内存，Prefill 整段长文本
- 避免分段处理和重载参数和KV

GitHub: https://github.com/lucienhuangfu/eLLM