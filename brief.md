eLLM 让纯 CPU 服务器也能跑出比 GPU 更快的 LLM 推理。  
eLLM 让 CPU（Xeon/EPYC）成为最佳的 AI 推理芯片。  
在 Prefill 为主的长文本推理任务中，单块 CPU （Xeon） 甚至可快过 8 块 GPU（H20）。
- "以存换算"重构纯 CPU 推理框架
- 依托 CPU 大内存，Prefill 整段长文本
- 避免分段处理和重载参数和KV

GitHub: https://github.com/lucienhuangfu/eLLM