---
hide:
  - navigation
  - toc
---

# 欢迎使用 eLLM

eLLM 是一个用 Rust 编写的高性能 CPU 推理引擎，提供 OpenAI 兼容的 HTTP 服务层——无需 GPU。

---

## 从哪里开始

- **立即运行模型** → [快速入门](getting_started/quickstart.md)
- **从源码构建** → [安装指南](getting_started/installation.md)
- **了解架构** → [设计文档](design/index.md)
- **新增算子** → [贡献指南](contributing/new_operator.md)

---

## 核心特性

- 纯 Rust 推理运行时，支持 AVX-512 / AVX2 SIMD 内核
- OpenAI 兼容的 `/v1/chat/completions` 端点，支持 SSE 流式输出
- 事件驱动批次调度器——阈值 + 超时双触发机制
- 支持 Dense 和 Sparse-MoE Transformer 架构（Qwen3、MiniMax-M2.5、Llama-2、Mixtral）
- 基于黄金文件的 HuggingFace 对齐工作流，用于算子验证
