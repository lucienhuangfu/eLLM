# 04 · 返回格式与 `/status` 接口

> 对应原 `serving.md` 章节：§5 返回格式、§6 `/status` 接口。

## 1. 非流式返回

返回 `ChatCompletionResponse`，结构上与 OpenAI chat completion 兼容：

* `id`：`chatcmpl-{nanos}`
* `object`：`"chat.completion"`
* `created`：Unix 时间戳（秒）
* `model`：请求中传入的 model 字段
* `choices[0].message`：`{role: "assistant", content: <生成文本>}`
* `choices[0].finish_reason`：`"stop"`

## 2. 流式返回

返回 SSE 响应，事件体为 `StreamResponse`（`object: "chat.completion.chunk"`）。

> 当前实现下，流式与 vLLM 一样是**逐 token 增量推送**（1 token / chunk）。详细的实现细节、字段复用与 vLLM 对比见 [06-streaming-comparison.md](./06-streaming-comparison.md)。

## 3. `GET /status`

`GET /status` 返回一个简单 JSON，用于健康检查：

```json
{
  "status": "running",
  "mode": "single_threaded_background_processing",
  "info": "Inference and HTTP server run on a single OS thread using current_thread runtime"
}
```
