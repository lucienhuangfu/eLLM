---
kind: external_dependency
name: Hugging Face 模型格式
slug: huggingface-transformers
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
source_files:
    - src/config/huggingface_config.rs
    - models/Qwen3-0.6B/
    - models/MiniMax-M2.5/
---

项目完全兼容 Hugging Face Transformers 的模型格式，包括 config.json、tokenizer.json、model.safetensors 等标准文件结构。支持从 Hugging Face Hub 下载的预训练模型，包括 Qwen3 和 MiniMax M2.5 系列模型。