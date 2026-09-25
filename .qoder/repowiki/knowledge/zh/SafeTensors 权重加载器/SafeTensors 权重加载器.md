---
kind: external_dependency
name: SafeTensors 权重加载器
slug: safetensors
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
source_files:
    - src/runtime/loader/safetensors.rs
    - src/runtime/init.rs
---

项目使用 safetensors 库来加载模型权重文件。这是一种安全的张量序列化格式，专门用于机器学习模型参数的存储和传输。项目在运行时通过 SafeTensorsLoader 从模型目录加载所有权重参数到内存中，支持 f16 精度格式。