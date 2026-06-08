# Serving 模块说明

---

`src/serving` 提供的是一层 OpenAI 兼容的 HTTP 服务封装。它本身不负责模型计算，而是负责：

* 接收 `/v1/chat/completions` 请求
* 为请求分配 batch 槽位
* 等待外部推理 runner 完成
* 按流式或非流式格式返回结果

---

## 1. 模块入口

`serving::run()` 是服务入口：

1. 启动 `token_counter.run()` 后台定时任务
2. 构建 `ApiState`（通过 `build_api_state()`）
3. 注册路由
4. 绑定 `0.0.0.0:8000`
5. 启动 Axum HTTP 服务

当前暴露的接口有两个：

* `POST /v1/chat/completions`
* `GET /status`

---

## 2. 状态结构

`ApiState` 维护了以下共享对象：

* `batch_sequences`：`Arc<SharedMut<BatchSequence<f16>>>`，持有 tokenizer 和 token 序列缓冲区
* `batch_states`：`Arc<SharedMut<Vec<SequenceState>>>`，每个 batch 槽位的推理状态
* `token_counter`：`Arc<TokenCounter>`，调度触发器
* `free_slots`：`Arc<Mutex<VecDeque<usize>>>`，空闲槽位队列
* `available_slots`：`Arc<Semaphore>`，并发控制信号量

启动时会扫描 `batch_states`，把 `Phase::Start` 的槽位放入空闲队列，并初始化信号量许可数。

信号量和队列主要用于槽位占用管理，防止超过 `batch_size` 的并发请求同时写入。

---

## 3. 请求处理流程

`chat_completions()` 的处理过程：

1. 生成请求 ID（纳秒时间戳格式：`chatcmpl-{nanos}`）
2. 读取 `stream` 参数，决定流式或非流式返回
3. 调用 `assign_slot_with_messages()` 申请空闲槽位
4. `notifier.notified().await` 异步等待推理完成
5. 从 `batch_states` 读取生成结果，调用 `decode_generated_text()` 解码
6. 调用 `reclaim_slot()` 释放槽位
7. 返回 OpenAI 风格响应

如果写入 prompt 失败（tokenization 错误），会直接返回 `500`。

---

## 4. 槽位分配逻辑

槽位分配由 `assign_slot_with_messages()` 完成：

1. 通过 `Semaphore` 获取 permit（背压控制，超过 `batch_size` 时阻塞等待）
2. 从 `free_slots` 队列弹出一个空闲槽位索引
3. 调用 `batch_sequences.write_prompts()` 渲染 chat template、tokenize 并写入序列缓冲区
4. 将槽位状态设为 `Phase::Prefill`；`sequence_index` 和 `kv_index` 初始化为 `0`，设置 `filling_length`
5. 调用 `permit.forget()` 将 permit 从 RAII 中分离，后续由 `reclaim_slot()` 手动归还
6. 调用 `token_counter.increment(write_len)` 触发调度
7. 返回 `(slot_index, notifier)`

槽位释放由 `reclaim_slot(state, slot_index, release_permit)` 完成：

* 状态重置为 `Phase::Start`
* `sequence_index`、`kv_index` 清零到哨兵值（`usize::MAX`）
* `filling_length` 归零
* 槽位重新放回 `free_slots` 队列
* 若 `release_permit` 为 `true`，调用 `available_slots.add_permits(1)` 手动归还一个信号量许可

---

## 5. 返回格式

### 非流式

返回 `ChatCompletionResponse`，结构上与 OpenAI chat completion 兼容：

* `id`：`chatcmpl-{nanos}`
* `object`：`"chat.completion"`
* `created`：Unix 时间戳（秒）
* `model`：请求中传入的 model 字段
* `choices[0].message`：`{role: "assistant", content: <生成文本>}`
* `choices[0].finish_reason`：`"stop"`

### 流式

返回 SSE 响应，事件体为 `StreamResponse`（`object: "chat.completion.chunk"`）。

实现上使用 `split_inclusive(' ')` 切分完整生成文本——每个 chunk 保留尾部空格——逐词输出，最后一个词附带 `finish_reason: "stop"`。当前流式返回是"分片发送最终文本"，不是边推理边增量生成 token。

---

## 6. `/status` 接口

`GET /status` 返回一个简单 JSON，用于健康检查：

```json
{
  "status": "running",
  "mode": "single_threaded_background_processing",
  "info": "Inference and HTTP server run on a single OS thread using current_thread runtime"
}
```

---

## 7. 服务初始化

`initialize_serving_resources(config)` 在 `resources.rs` 中完成所有组件的初始化，按顺序：

1. 加载模型配置（`config.json`）和生成配置（`generation_config.json`）
2. 通过 `SafeTensorsLoader` 加载权重，写入全局内存池
3. 提取采样参数（top_k、top_k_simd、top_p、min_p、do_sample、eos_token_id_list）
4. 确定线程配置（worker_threads = max(total - async_threads, 1)，async_threads = 2）
5. 构建 `BatchSequence`（tokenizer + 序列缓冲区）
6. 构建 `batch_states`（每个槽位的 `SequenceState`）
7. 创建调度组件（`BatchScheduler` + `TokenCounter` + broadcast channel）
8. 创建 RoPE 位置编码，初始化模型，执行一次前向推理（填充算子队列）
9. 创建 `ServingRunner`（从全局队列取出算子列表）

返回 `ServingResources`，包含所有运行时组件。

---

## 8. 当前实现特点

* HTTP 层是 OpenAI 兼容风格
* 推理由外部 `ServingRunner` 驱动，serving 层本身不执行模型计算
* 槽位管理通过 `Semaphore + VecDeque` 实现背压控制
* `temperature` 已在请求体中支持，写入 `batch_temperature` 参与采样
* `max_tokens`、`top_p` 已在请求体中保留，但当前 handler 中未参与调度逻辑

---

## 9. 代码入口参考

* `src/serving/mod.rs` — HTTP 服务器入口、路由、API 数据结构
* `src/serving/config.rs` — `ServingConfig`（环境变量读取）
* `src/serving/resources.rs` — `ServingResources` 整合初始化
* `src/serving/model_setup.rs` — 模型加载、参数提取、线程配置
* `src/serving/model.rs` — 模型初始化与前向推理封装
* `src/serving/scheduler.rs` — 调度组件创建
* `src/serving/chat_handlers.rs` — `chat_completions` HTTP handler
* `src/serving/parser.rs` — 模型输出增量流式解析器

---

## 10. 流式实现与 vLLM 的对比

### vLLM 的流式粒度

vLLM 的流式是**每生成 1 个 token 就发送一次 SSE chunk**，即 1 token/chunk。推理循环每 decode 出一个 token 就立刻推送给客户端，首 token 延迟（TTFT）极低，是真正的增量流式。

### 当前 eLLM 的流式实现

eLLM 复用 `SequenceState` 中已有的 `notify: Arc<Notify>`，不引入任何新字段，实现真增量流式：

**推理侧（`TopKSoftmax::run`）**

每次成功写入一个 decode token 后（包括 EOS token），立即调用 `record.notify.notify_one()`。EOS 时先将 `phase` 设为 `Phase::Eos`，再 notify。

**服务侧（`chat_handlers::build_stream_response`）**

流式路径不再等待一次性完成通知，而是进入循环：

1. `notifier.notified().await` — 等待下一个 token
2. 读取 `record.sequence_index`（topk 写入后设为当前 token 的位置）和 `record.phase`
3. 调用 `batch_sequences.decode_single_token(slot_index, token_index)` 解码单个 token
4. 立即推送一条 SSE chunk
5. 若 `phase == Eos`，发送带 `finish_reason: "stop"` 的最后一个 chunk，退出循环
6. 循环结束后调用 `reclaim_slot()` 释放槽位

**非流式路径不变**：仍然等待 EOS 的单次 notify，然后一次性 decode 全部生成文本。

### 与 vLLM 的对比

| | vLLM | eLLM（当前） |
|---|---|---|
| 流式粒度 | 1 token/chunk | 1 token/chunk |
| 新增字段 | 无 | 无（复用 `Notify`） |
| TTFT | 极低 | 极低 |
| 非流式路径 | 不变 | 不变 |

---

## 11. 输出解析器（`parser.rs`）

`src/serving/parser.rs` 实现了一个增量流式解析器，用于解释模型文本输出，从中提取普通内容、推理链片段和工具调用，**且从不重新扫描历史输出**。

### 设计目标

* **零重解析** — 每次只处理新到达的增量文本，已发出的事件不会再次处理。
* **边界安全缓冲** — 当缓冲区尾部可能是某个标签的前缀时，自动保留该尾部，防止跨 chunk 的标签被漏掉。
* **加载时规则选择** — 格式规则在构造时由 `ModelFamily` 决定，运行时不再推断。

---

### 核心类型

#### `ParserRule`

保存某个模型族的四个标签字符串及工具调用格式：

| 字段 | 作用 |
|---|---|
| `tool_start` / `tool_end` | 工具调用块的开启/关闭标签 |
| `think_start` / `think_end` | 思维链推理块的开启/关闭标签 |
| `tool_format` | 如何解析工具块内部的 payload |

内置预设与对应的模型族：

| 预设 | `ModelFamily` | 工具格式 |
|---|---|---|
| `qwen()` | `Qwen` | `Tagged` |
| `deepseek()` | — | `Tagged` |
| `hermes()` | 兜底 / `Unknown` | `Tagged` |
| `llama3_json()` | `Llama` | `RawJson` |
| `mistral()` | `Mixtral` | `PrefixedJson` |
| `minimax_m1()` | `MiniMax`、`MiniMaxM2` | `Tagged` |
| `minimax_m2()` | — | `MiniMaxM2` |

`ParserRule::for_model_family(family)` 会自动选择正确的预设。

#### `ToolCallFormat`

控制工具块内部 JSON payload 的解码方式：

* **`Tagged`** — 块内直接是 JSON 对象（或数组），用 `serde_json` 解析。
* **`PrefixedJson`** — Mistral 风格：`[TOOL_CALLS] FunctionName { ... }`，函数名在 JSON 体之前。
* **`RawJson`** — Llama 3 风格：`<|python_tag|>` 哨兵符之后是原始 JSON 对象，无闭合标签。
* **`MiniMaxM2`** — 类 XML 的 `<invoke name="…"><parameter name="…">value</parameter></invoke>` 语法。

#### `ParserOptions`

在 `ParserRule` 基础上增加两个功能开关：

* `reasoning_parser` — 为 `false` 时，`<think>…</think>` 标签直接作为 `Content` 透传。
* `tool_call_parser` — 为 `false` 时，工具调用标签直接作为 `Content` 透传。

两个开关在 `ParserOptions::new(rule)` 中默认均为 `true`。

#### `ParserEvent`

每次 `feed()` 返回一个 `Vec<ParserEvent>`：

| 变体 | 含义 |
|---|---|
| `Content(String)` | 显示给用户的普通助手文本 |
| `Reasoning(String)` | 思维链片段（可隐藏或单独展示） |
| `ToolCallDelta(ToolCallDelta)` | 正在生成的工具调用的原始 JSON 片段 |
| `ToolCall(ToolCall)` | 已完成并解析好的工具调用（含 `name` 和 `arguments`） |
| `Finish` | （保留）流结束信号 |

`ToolCall` 包含 `name: String` 和 `arguments: serde_json::Value`。

---

### 状态机

`IncrementalStreamingParser` 驱动一个四状态机：

```
Normal  ──think_start──►  Reasoning  ──think_end──►  Normal
Normal  ──tool_start───►  ToolCall   ──（自动）────►  ToolJson
ToolJson ──tool_end────►  Normal
```

| 状态 | 每次 `feed()` 的行为 |
|---|---|
| `Normal` | 将内容发射为 `Content`，直到遇到下一个标记；保留可能是标签前缀的尾部 |
| `Reasoning` | 将片段发射为 `Reasoning`，直到找到 `think_end` |
| `ToolCall` | 标记符已消耗，立即跳转到 `ToolJson` |
| `ToolJson` | 累积 JSON，发射 `ToolCallDelta` 片段，完整解析后发射 `ToolCall` |

辅助函数 `longest_suffix_prefix_len()` 计算缓冲区尾部有多少字节是某个标签字符串的前缀——正是这一机制让解析器在不缓冲整个流的前提下做到边界安全。

---

### 工具调用 payload 解析

四种内部策略对应四种 `ToolCallFormat`：

**`Tagged` / `MiniMaxM2`**  
累积 `tool_start` 和 `tool_end` 之间的所有字节。遇到 `tool_end` 时：
- `Tagged`：调用 `serde_json::from_str` + `extract_tool_call_value`，支持扁平 `{name, arguments}` 和嵌套 `{function: {name, arguments}}` 两种对象形式，以及顶层 JSON 数组形式。
- `MiniMaxM2`：调用 `parse_minimax_m2_tool_call`，用字符串扫描（不依赖 XML 解析库）遍历类 XML 的 `<invoke>` / `<parameter>` 结构。

**`RawJson`**  
由 `<|python_tag|>`（无闭合标签）触发。将后续所有文本缓冲，用 `extract_complete_json_prefix_range` 检测完整 JSON 值何时到达。JSON 对象之后的剩余部分被推回主缓冲区，以 `Normal` 状态继续解析。

**`PrefixedJson`**  
由 `[TOOL_CALLS]` 触发。期望格式为 `FunctionName { … }`——第一个 `{` 之前的文本作为函数名，`extract_complete_json_prefix_range` 提取 JSON 体，两者合并为一个 `ToolCall`。

---

### `parser_event_stream`

一个便捷适配器，将 `StreamingParser` 和 `Stream<Item = TextDelta>` 组合成 `Stream<Item = ParserEvent>`：把每个 delta 喂给解析器，将产生的事件逐一 yield 出去。设计用于 `chat_handlers` 的流式响应路径。

```rust
let events = parser_event_stream(parser, delta_stream);
```

---

### 运行时禁用解析器

推理解析器和工具调用解析器可以独立关闭：

```rust
let options = ParserOptions {
    rule: ParserRule::qwen(),
    reasoning_parser: false,  // <think> 标签作为 Content 透传
    tool_call_parser: false,  // <tool_call> 标签作为 Content 透传
};
let mut parser = IncrementalStreamingParser::with_options(options);
```

这在下游消费者需要原始模型输出，或部署的模型不使用标准标签时非常有用。
