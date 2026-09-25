---
kind: external_dependency
name: Tokio 异步运行时
slug: tokio
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
source_files:
    - src/bin/main.rs
---

项目使用 Tokio 作为异步运行时环境，提供多线程异步执行能力。主程序通过 tokio::runtime::Builder 创建多线程运行时，支持 API 线程和阻塞线程的分离配置。Tokio 还用于网络 I/O、任务调度等核心异步操作。