# 安装

## 依赖要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Rust 工具链 | 1.78+ | 通过 [rustup](https://rustup.rs) 安装 |
| CPU | 支持 AVX-512 的 x86-64 | 也提供 AVX2 标量降级路径 |
| 操作系统 | Linux 或 Windows | 已在 Ubuntu 22.04 和 Windows 11 上测试 |
| 内存 | 8 GB+ | 取决于模型大小 |

> **注意：** eLLM 是纯 CPU 推理引擎，不需要 GPU。

---

## 从源码构建

```bash
git clone <repo-url> eLLM
cd eLLM

# 调试构建（编译快，推理慢）
cargo build

# Release 构建（推荐用于 serving）
cargo build --release
```

Release profile 开启了 `opt-level = 3`、`lto = fat`、`codegen-units = 1`，
可获得最大单线程吞吐量。

---

## 下载模型

将 HuggingFace 兼容的模型目录放到 `models/` 下。最少需要以下文件：

```
models/<model-name>/
├── config.json
├── generation_config.json
└── model.safetensors
```

serving 路径还需要 tokenizer：

```
models/<model-name>/
├── tokenizer.json
└── tokenizer_config.json
```

目前已测试的模型：

| 模型 | 目录名 |
|------|-------|
| Qwen3-0.6B | `models/Qwen3-0.6B` |
| MiniMax-M2.5 | `models/MiniMax-M2.5` |
| Llama-2-7B | `models/Llama-2-7b-hf` |
| Llama-2-70B | `models/Llama-2-70b-hf` |

---

## 环境变量

所有运行时参数通过环境变量控制。完整列表见[环境变量](../reference/env_vars.md)。

快速默认值：

```bash
export ELLM_BATCH_SIZE=3
export ELLM_SEQUENCE_LENGTH=128
export ELLM_CHUNK_SIZE=64
export ELLM_SCHEDULE_TIMEOUT_MS=10
```

---

## 启动服务

```bash
# 在 8000 端口服务 Qwen3-0.6B
cargo run --release --bin main -- --model models/Qwen3-0.6B

# 或者使用 backend 二进制
cargo run --release --bin backend
```

服务绑定到 `0.0.0.0:8000`，暴露：

- `POST /v1/chat/completions`
- `GET /status`

示例请求见[快速入门](./quickstart.md)。
