# Serving 模块

`src/serving` 提供的是一层 OpenAI 兼容的 HTTP 服务封装。它本身不负责模型计算，而是负责：

* 接收 `/v1/chat/completions` 请求
* 为请求分配 batch 槽位
* 等待外部推理 runner 完成
* 按流式或非流式格式返回结果

## 目录

| 文档 | 主题 |
| --- | --- |
| [01-overview.md](./01-overview.md) | 模块入口、当前实现特点、代码入口参考 |
| [02-state-and-slot.md](./02-state-and-slot.md) | `ApiState` 状态结构、槽位分配与释放 |
| [03-request-flow.md](./03-request-flow.md) | `/v1/chat/completions` 请求处理流程 |
| [04-response-format.md](./04-response-format.md) | 非流式/流式返回格式、`/status` 接口 |
| [05-initialization.md](./05-initialization.md) | `initialize_serving_resources` 初始化流程 |
| [06-streaming-comparison.md](./06-streaming-comparison.md) | 增量流式实现以及与 vLLM 的对比 |

## 与原 `serving.md` 的章节对应

| 新文件 | 原 `serving.md` 章节 |
| --- | --- |
| `01-overview.md` | §1 模块入口、§8 当前实现特点、§9 代码入口参考 |
| `02-state-and-slot.md` | §2 状态结构、§4 槽位分配逻辑 |
| `03-request-flow.md` | §3 请求处理流程 |
| `04-response-format.md` | §5 返回格式、§6 `/status` 接口 |
| `05-initialization.md` | §7 服务初始化 |
| `06-streaming-comparison.md` | §10 流式实现与 vLLM 的对比 |

## 当前暴露的接口

* `POST /v1/chat/completions`
* `GET /status`
