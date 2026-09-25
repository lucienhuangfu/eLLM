---
kind: external_dependency
name: TikTok Tokenizer Library
slug: tiktoken-rs
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
source_files:
    - Cargo.toml
---

项目使用 tiktoken-rs 作为分词器实现，用于处理 Qwen3 系列模型的 tokenization。该库是 OpenAI 的 tiktoken 的 Rust 实现，支持 BPE 分词算法。在项目中主要用于模型配置加载和文本预处理阶段，与 HuggingFace 格式的模型配置文件配合使用。