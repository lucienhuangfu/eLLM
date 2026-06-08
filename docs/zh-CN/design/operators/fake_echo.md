# Fake Echo Operator 文档

这份文档说明 `FakeEcho` 的用途、执行条件和当前实现行为。

对应代码：

- `src/operators/fake_echo.rs`
- `src/runtime/operator.rs`
- `src/bin/fake_server.rs`

---

## 1. 它是做什么的

`FakeEcho` 是一个专门用于联调和集成测试的测试算子。

它的目标不是完成真实的模型计算，而是证明这条 runtime 链路确实执行过一个 operator，并且这个 operator 对请求状态产生了可见影响。

在 `fake_server` 场景里，它可以替代真实模型的复杂前向过程，让你快速验证下面这些环节：

1. `ServingRunner` 是否能正常调度 operator queue
2. `batch_list` 的状态是否会从 `Prefill` 推进到 `Eos`
3. 请求等待中的 `Notify` 是否会被正确唤醒

---

## 2. 执行条件

`FakeEcho` 只在 `thread_id == 0` 时执行。

这是因为当前 `ServingRunner` 的 thread 0 承担调度推进职责，而 `FakeEcho` 也沿用了这个约定：

- 非 0 号线程直接返回
- 只有 0 号线程会遍历 `batch_list`

这样做的好处是：

1. 行为确定
2. 便于调试
3. 不需要额外的跨线程同步逻辑

---

## 3. 当前行为

`FakeEcho` 只处理 `Phase::Prefill` 的记录。

如果某个 slot 处于 `Prefill`，它会做三件事：

1. 读取该 slot 的 `filling_length`
2. 将该 slot 推进到 `Phase::Eos`
3. 唤醒等待中的请求

### 3.1 状态变化

在完成处理后，`FakeEcho` 会把 slot 状态改成：

- `sequence_index = 0`
- `kv_index = filling_length`
- `filling_length = 0`
- `phase = Phase::Eos`

然后调用 `notify.notify_one()` 唤醒等待请求。

---

## 4. 和真实 serving 的关系

`FakeEcho` 主要服务于 `src/bin/fake_server.rs`。

`fake_server` 的结构是：

1. 创建 `BatchSequence`
2. 创建 `BatchScheduler`
3. 把 `FakeEcho` 放进 `ServingRunner` 的 operator queue
4. 启动 runtime 线程和 HTTP 服务

这意味着请求进入后，虽然不会经过真实的模型前向，但仍然会经历完整的 serving 生命周期：

1. 请求写入 token buffer
2. slot 进入 `Prefill`
3. `ServingRunner` 调度到 `FakeEcho`
4. `FakeEcho` 结束请求
5. HTTP 层读取生成结果并返回

---

## 5. 适合什么场景

`FakeEcho` 适合下面这些用途：

1. 验证 serving 和 runtime 的接线是否通
2. 检查 `Notify`、`Phase` 和 slot 回收逻辑
3. 调试请求生命周期
4. 做不依赖真实模型权重的最小联调

它不适合做数值正确性验证，也不适合替代真实 operator 的单测。

---

## 6. 代码入口

### 6.1 算子定义

`src/operators/fake_echo.rs` 中定义了 `FakeEcho` 本体。

### 6.2 Runtime 分发

`src/runtime/operator.rs` 中通过 `Operator::FakeEcho` 分发到 `FakeEcho::run()`。

### 6.3 Fake Server

`src/bin/fake_server.rs` 会把 `FakeEcho` 放进 operator queue，让它在 runtime 中直接推进 `Prefill` 请求结束。

---

## 7. 设计注意点

当前实现是一个“测试友好”的 fake operator，不是通用业务算子。

需要注意：

1. 它只关注 `Prefill`，不会实现真实 decode
2. 它不会直接写 token buffer，因此更适合做请求生命周期联调
3. 它的行为是刻意简化的，目的是“看得出来运行过”，不是“像真实模型一样生成文本”
