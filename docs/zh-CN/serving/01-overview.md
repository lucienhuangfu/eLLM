# 01 · 模块入口、当前实现特点、代码入口参考

> 对应原 `serving.md` 章节：§1 模块入口、§8 当前实现特点、§9 代码入口参考。

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

详见 [04-response-format.md](./04-response-format.md) 的接口说明。

## 2. 当前实现特点

* HTTP 层是 OpenAI 兼容风格
* 推理由外部 `ServingRunner` 驱动，serving 层本身不执行模型计算
* 槽位管理通过 `Semaphore + VecDeque` 实现背压控制
* `temperature` 已在请求体中支持，写入 `batch_temperature` 参与采样
* `max_tokens`、`top_p` 已在请求体中保留，但当前 handler 中未参与调度逻辑

## 3. 代码入口参考

* `src/serving/mod.rs` — HTTP 服务器入口、路由、API 数据结构
* `src/serving/config.rs` — `ServingConfig`（环境变量读取）
* `src/serving/resources.rs` — `ServingResources` 整合初始化
* `src/serving/model_setup.rs` — 模型加载、参数提取、线程配置
* `src/serving/model.rs` — 模型初始化与前向推理封装
* `src/serving/scheduler.rs` — 调度组件创建
* `src/serving/chat_handlers.rs` — `chat_completions` HTTP handler
