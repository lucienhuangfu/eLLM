# 环境变量

所有 eLLM 运行时参数通过环境变量配置，在 `ServingConfig::new()` 中读取
（`src/serving/config.rs`）。值为 `0` 时回退到内置默认值。

---

## Serving / 调度

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ELLM_BATCH_SIZE` | `3` | 最大并发飞行中请求数（batch 槽位数） |
| `ELLM_SEQUENCE_LENGTH` | `128` | 每个槽位的最大 token 序列长度。内存线性增长。 |
| `ELLM_CHUNK_SIZE` | `64` | 每轮 prefill 最多处理的 token 数；同时用作调度 `token_threshold`。 |
| `ELLM_SCHEDULE_TIMEOUT_MS` | `10` | 超时窗口（毫秒），即使未达到阈值，当 `token_counter > 0` 时也触发调度。 |

### 调优策略

| 目标 | 调整方式 |
|------|---------|
| 降低延迟 | 减小 `ELLM_CHUNK_SIZE` |
| 提高吞吐 | 增大 `ELLM_CHUNK_SIZE` 和 `ELLM_BATCH_SIZE` |
| 更长上下文 | 增大 `ELLM_SEQUENCE_LENGTH` |
| 突发 / 低流量 | 减小 `ELLM_SCHEDULE_TIMEOUT_MS` |

---

## 线程数

线程数由 `src/serving/model_setup.rs` 中的 `determine_thread_config()` 自动计算，
不能直接通过环境变量配置。

| 线程池 | 数量 | 计算公式 |
|--------|------|---------|
| Worker 线程 | `max(total_cpus - async_threads, 1)` | CPU 密集型 |
| Async 线程 | `2` | Tokio 异步执行器 |

---

## 模型路径

模型目录通常通过 CLI 参数（`--model <path>`）传给主二进制。
入口见 `src/bin/main.rs` 或 `src/bin/backend.rs`。
