# Serving 模块说明

---

`src/serving` 提供的是一层 OpenAI 兼容的 HTTP 服务封装。它本身不负责模型计算，而是负责：

* 接收 `/v1/chat/completions` 请求
* 为请求分配 batch 槽位
* 等待外部推理 runner 完成
* 按流式或非流式格式返回结果

当前实现明确是单线程驱动的，不做多线程并发写入。这里的设计动机不是单纯追求“简单”，而是单线程已经足以满足当前需求，因此 HTTP 只负责串行接收、写入和返回，不需要额外引入多线程同步开销。

---

## 1. 模块入口

`serving::run()` 是服务入口：

1. 初始化 `AppState`
2. 注册路由
3. 绑定 `0.0.0.0:8000`
4. 启动 Axum HTTP 服务

整个 serving 侧运行在单线程背景处理模型下，因此请求处理和槽位写入都按串行方式推进。

当前暴露的接口有两个：

* `POST /v1/chat/completions`
* `GET /status`

---

## 2. 状态结构

`AppState` 维护了两个核心共享对象：

* `batch_sequences`
* `batch_list`

另外还维护了槽位管理结构：

* `free_slots`：空闲槽位队列
* `available_slots`：信号量，用于控制可用槽位数量

启动时会扫描 `batch_list`，把 `Phase::Start` 的槽位放入空闲队列。

这里的信号量和队列主要是为了做槽位占用管理，不是为了把写入过程并行化；实际写入仍然是单线程完成的，因为这已经足以满足当前服务场景。

---

## 3. 请求处理流程

`chat_completions()` 的处理过程可以概括为：

1. 生成请求 ID
2. 读取 `stream` 参数，决定流式或非流式返回
3. 申请一个空闲槽位
4. 把请求消息写入 batch 对应位置
5. 将该槽位状态切到 `Phase::Prefill`
6. 等待外部推理完成通知
7. 读取生成文本并释放槽位
8. 返回 OpenAI 风格响应

如果写入 prompt 失败，会直接返回 `500`。

---

## 4. 槽位分配逻辑

槽位分配由 `assign_slot_with_messages()` 完成：

* 先从信号量获取许可
* 再从 `free_slots` 队列弹出一个索引
* 然后写入消息并更新 `SequenceState`

成功后会把 `Notify` 返回给 handler，用于等待该槽位的推理完成。

释放槽位由 `reclaim_slot()` 完成：

* 状态重置为 `Phase::Start`
* `sequence_index`、`kv_index` 清零到哨兵值
* `filling_length` 归零
* 槽位重新放回空闲队列
* 需要时归还信号量许可

---

## 5. 返回格式

### 非流式

返回 `ChatCompletionResponse`，结构上与 OpenAI chat completion 类似：

* `id`
* `object = "chat.completion"`
* `created`
* `model`
* `choices[0].message`

### 流式

返回 SSE 响应，事件体为 `StreamResponse`。

实现上会把完整生成文本按空格切分后逐段输出，因此当前流式返回更像是“分片发送最终文本”，不是边推理边增量生成 token。

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

## 7. 当前实现特点

* HTTP 层是 OpenAI 兼容风格
* 请求处理和 batch 写入都是单线程串行执行
* 服务端本身不执行模型推理
* 推理由外部 runner 驱动
* 当前是单线程背景处理模式
* `temperature`、`max_tokens`、`top_p` 已在请求体中保留，但在当前 handler 中未参与调度逻辑

---

## 8. 代码入口参考

* `src/serving/mod.rs`
* `src/serving/bootstrap.rs`
* `src/serving/handlers.rs`
* `src/serving/types.rs`
