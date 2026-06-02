# 02 · 状态结构与槽位分配

> 对应原 `serving.md` 章节：§2 状态结构、§4 槽位分配逻辑。

## 1. `ApiState` 共享对象

`ApiState` 维护了以下共享对象：

* `batch_sequences`：`Arc<SharedMut<BatchSequence<f16>>>`，持有 tokenizer 和 token 序列缓冲区
* `batch_states`：`Arc<SharedMut<Vec<SequenceState>>>`，每个 batch 槽位的推理状态
* `token_counter`：`Arc<TokenCounter>`，调度触发器
* `free_slots`：`Arc<Mutex<VecDeque<usize>>>`，空闲槽位队列
* `available_slots`：`Arc<Semaphore>`，并发控制信号量

启动时会扫描 `batch_states`，把 `Phase::Start` 的槽位放入空闲队列，并初始化信号量许可数。

信号量和队列主要用于槽位占用管理，防止超过 `batch_size` 的并发请求同时写入。

## 2. 槽位分配

槽位分配由 `assign_slot_with_messages()` 完成：

1. 通过 `Semaphore` 获取 permit（背压控制，超过 `batch_size` 时阻塞等待）
2. 从 `free_slots` 队列弹出一个空闲槽位索引
3. 调用 `batch_sequences.write_prompts()` 渲染 chat template、tokenize 并写入序列缓冲区
4. 将槽位状态设为 `Phase::Prefill`；`sequence_index` 和 `kv_index` 初始化为 `0`，设置 `filling_length`
5. 调用 `permit.forget()` 将 permit 从 RAII 中分离，后续由 `reclaim_slot()` 手动归还
6. 调用 `token_counter.increment(write_len)` 触发调度
7. 返回 `(slot_index, notifier)`

## 3. 槽位释放

槽位释放由 `reclaim_slot(state, slot_index, release_permit)` 完成：

* 状态重置为 `Phase::Start`
* `sequence_index`、`kv_index` 清零到哨兵值（`usize::MAX`）
* `filling_length` 归零
* 槽位重新放回 `free_slots` 队列
* 若 `release_permit` 为 `true`，调用 `available_slots.add_permits(1)` 手动归还一个信号量许可
