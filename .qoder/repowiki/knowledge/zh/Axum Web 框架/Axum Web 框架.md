---
kind: external_dependency
name: Axum Web 框架
slug: axum
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
source_files:
    - src/serving/server.rs
    - src/serving/types.rs
---

项目使用 Axum 作为 HTTP 服务器框架，提供 OpenAI 兼容的 RESTful API 接口。Axum 基于 Tokio 异步运行时，支持流式响应和并发请求处理。项目实现了 ChatCompletionRequest/Response 等标准 OpenAI API 结构体。