---
kind: error_handling
name: eLLM 错误处理体系：分层枚举 + anyhow/thiserror + axum 响应映射
category: error_handling
scope:
    - '**'
source_files:
    - src/config/config_validator.rs
    - src/serving/error.rs
    - src/runtime/loader/safetensors.rs
    - src/serving/server.rs
    - src/bin/main.rs
    - Cargo.toml
---

## 1. 采用的错误处理系统
- **配置校验层**：使用 `thiserror` 定义强类型、带用户可读消息的枚举错误，集中用于 YAML/JSON 配置解析与业务规则校验。
- **运行时 I/O 层**：在 safetensors 权重加载等外部依赖场景统一使用 `anyhow::Result`，以字符串化错误向上冒泡到调用方。
- **HTTP 服务层**：自定义 `ApiError` 枚举实现 `axum::IntoResponse`，将内部错误转换为 HTTP 状态码与 JSON 文本；上层 handler 通过 `match` 分支把具体错误转为响应。
- **二进制入口**：各 `bin/*` 程序普遍返回 `Result<(), Box<dyn std::error::Error>>`，由 Rust 运行时打印 backtrace 后退出。
- **panic 策略**：`Cargo.toml` 中 release profile 设置 `panic = "abort"`，生产构建直接终止进程而非展开栈；测试与对齐脚本中大量使用 `.unwrap()`，属于开发期快速失败模式。

## 2. 关键文件与包
- `src/config/config_validator.rs` — `ConfigError` 枚举（thiserror），覆盖 model/scheduler/serve/command 段校验。
- `src/serving/error.rs` — `ApiError` 枚举 + `ApiResult<T>` 别名 + `IntoResponse` 实现，统一 API 错误到 HTTP 响应。
- `src/runtime/loader/safetensors.rs` — 权重加载器，全面使用 `anyhow::{anyhow, Result}` 包装底层 I/O 与格式错误。
- `src/serving/server.rs` — Axum router 与 handler，演示如何将 `ApiError` 转成响应并释放 session slot。
- `src/bin/main.rs` / `backend.rs` / `fake_server.rs` — 二进制入口，统一返回 `Box<dyn std::error::Error>`。
- `Cargo.toml` — 声明 `anyhow`、`thiserror`、`axum` 依赖及 `panic = "abort"` 发布策略。

## 3. 架构与约定
- **按模块划分错误类型**：每个子系统维护自己的错误枚举，不共享跨层错误类型。配置用 `ConfigError`，HTTP 用 `ApiError`，I/O 用 `anyhow`。
- **错误向上传播路径**：底层 `anyhow::Result` → 中间层 `ConfigError` / `ApiError` → 顶层 `Box<dyn Error>` 或 `axum::Response`，避免在高层混用多种错误类型。
- **HTTP 错误映射规范**：当前所有 `ApiError` 变体均映射为 `500 Internal Server Error`，并通过 `eprintln!` 输出详细日志；未来可按语义区分 4xx/5xx。
- **panic 仅用于不可恢复状态**：release 下 abort 行为配合单元测试中的 `unwrap()`，保证异常快速暴露；生产环境不捕获 panic。

## 4. 开发者应遵循的规则
- **新增领域错误时优先定义枚举**：参考 `ConfigError`，使用 `#[derive(Error)]` 与 `#[error("...")]` 提供人类可读消息，便于日志与调试。
- **对外部 I/O 使用 `anyhow::Result`**：如 safetensors 加载、文件读取、网络请求，不要过早装箱为具体错误类型。
- **Axum handler 内统一转换**：在 handler 中对 `Err(e)` 分支调用 `e.into_response()`，保持路由层只关心成功路径。
- **避免在生产代码中使用 `unwrap()/expect()`**：对齐测试与示例脚本可保留，但核心库与服务逻辑应显式处理 `Result`。
- **谨慎使用 panic**：仅在断言不变量或数据损坏时使用；release 下会直接 abort，需配合监控告警。