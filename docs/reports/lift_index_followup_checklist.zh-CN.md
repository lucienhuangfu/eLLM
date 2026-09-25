# lift_index 保留核对与后续调整清单

审查基准：合并提交 `d3c1c11`；来源分支 `lift_index@4301f1f`。本轮只补充审查和清单，不继续修改功能、不以测试全绿作为合并条件。原 lift_index 分支未改写。

## 必须保留的设计

后续修复继续沿用 SlotSequence、SlotManager、Scheduler、ExecutorPool、SequenceSlice 和 lift_index，不恢复旧 Runner/BatchSequence/双 prefill/decode 列表架构。修复索引、并发和边界问题，应在这些模块内完成。

| 核对范围 | 与 lift_index 的对比 |
| --- | --- |
| src/runtime/scheduler（3 文件） | 完全一致；包括两遍调度、decode 优先预算、连续计算行布局、lift_index 分配 |
| src/runtime/session（4 文件） | 完全一致；包括 SlotSequence、会话复用、LRU/超时回收 |
| src/runtime/executor（3 文件） | 完全一致；包括 ExecutorPool、等待和 barrier |
| src/serving（11 文件） | 完全一致；包括 server、增量 parser、SSE、测试 |
| src/config（6 文件） | 完全一致；包括 CLI 和配置类型 |
| LookupRMSMap、SequenceSlice / ScheduleTask | 保留 lift_index 实现；total_size、lift_size 和三个索引字段保留 |

核对方法：`git diff --name-only lift_index d3c1c11 -- <目录>`；上述五个目录均无输出，共 27 个文件保持原样。

**并非整个合并树与 lift_index 完全相同。** 已有适配包括：保留 develop 的 CPU/MoE 优化和 aligned 权重加载；Experts→Expert 单数命名；TopKSoftmax 模块移位；LiftVector 改为单线程顺序压缩；普通/最后一层算子按 total_size/lift_size 分工；线程数减法防下溢。详见[合并报告](lift_index_merge_report.zh-CN.md)。这些差异需要后续验收，不能因为属于性能优化就默认正确。

## 状态与优先级

- **已观察**：已有测试结果证实失败。
- **静态确认**：代码中直接存在该行为，但本轮未运行完整触发场景。
- **待复现风险**：根据调用链或线程交错推导，需要定向测试确认后处理。
- P0：真实模型/并发上线前优先处理；P1：功能或资源管理问题；P2：兼容、性能和文档完善。以下条目均未关闭。

## P0：索引与并发

- [ ] **IDX-01：decode 输入位置与写入游标混用**（待复现风险）。位置：`src/operators/topk_softmax.rs:203`、`src/runtime/scheduler/scheduler.rs:147`、`src/operators/normalization/lookup_rms_map.rs:99`。Softmax 写入 sequences[N] 后把 record.next_sequence_index 变为 N+1；调度器下一轮直接传 N+1，Lookup 从该位置读，而刚生成的 token 在 N。应明确 next_sequence_index 是“待计算位置”还是“下次写入位置”，保持 lift_index/计算行概念不变。验收：prompt 长度 3 时，生成 token 写入第 3 格，下一轮读取第 3 格、KV/RoPE 位置也为 3，连续两轮与单步参考一致；最后一格不能跨 slot 读取。FakeEcho 不读模型 embedding，不能替代此验证。

- [ ] **CON-01：SharedMut 不提供同步与互斥**（静态确认，运行后果待定向复现）。位置：`src/operators/send_sync_ptr.rs:31`、`src/runtime/executor/executor_pool.rs:135`、`src/runtime/session/manager.rs`。SharedMut::with_mut 从 &self 构造 &mut T，多 worker 为同一个 Vec<SlotState> 创建可变引用，服务线程也直接读写；算子间 barrier 不等于服务线程和 worker 的同步。调整：为槽状态建立明确的所有权/锁或原子状态发布机制，worker 只获得独占切片；保留新调度架构。验收：并发读写/接入/释放可重复压力测试，适合的缩小模型通过 Miri 或竞态检测，明确每个 unsafe 的同步约定。

- [ ] **CON-02：任务发布、清空和下一轮启动缺少独立代次协议**（待复现风险）。位置：`src/runtime/executor/executor_pool.rs:85`、`src/runtime/scheduler/scheduler.rs:41`。其他线程轮询普通 task 字段，thread 0 同时 reset/重建；第二道 barrier 后非零线程可能看到尚未清空的上一轮任务并进入下一轮 barrier，尤其最后一个请求结束时可能卡住。调整：保留双遍调度，增加原子 epoch/ready 状态和停止协议。验收：反复执行“单请求结束→空闲→新请求”，不同 worker 延迟下不重复执行、不漏任务、不死锁。

- [ ] **SES-01：同 session 并发请求的错误清理可释放他人的活跃槽**（待复现风险，调用链可见）。位置：`src/runtime/session/manager.rs:54`、`src/serving/server.rs:377`。acquire 对活跃 session 返回同一 handle；第二个请求 write_prompts 因槽忙失败后，handler 调用 release_session，可能回收第一个请求仍在用的槽。调整：handle 携带请求租约/代次，只有持有者能释放；忙请求不取得释放权限。验收：同 session 两个重叠请求，第二个被拒绝或排队，第一请求生成内容和槽归属保持正确。

- [ ] **SES-02：延迟回收与重用存在时间窗口**（待复现风险）。位置：`src/runtime/session/manager.rs:112`。release 在移除 session_map 后才单独锁 reserved 插入；timeout 在锁 reserved 之前检查 cancel_flag，随后按 session_id 删除，未校验所删项的代次。可能出现重复占槽、旧 timer 删除新 reservation。调整：原子转移会话状态，timer 必须校验 slot+generation。验收：在旧超时触发前后反复 acquire/release 同 session，旧 timer 永远不回收新租约。

## P1：功能和容量

- [ ] **CACHE-01：全前缀命中没有剩余 prefill 时无法前进**（静态推导，待端到端复现）。位置：`src/runtime/session/manager.rs:163`、`src/runtime/scheduler/scheduler.rs:78`。完全命中时 next_sequence_index=prompt_length 且 phase=Prefill；若无其他 decode，schedule_batch 因 token 总数为零返回无工作；若有其他任务，可能生成 length=0 的末块切片，而 LiftVector 会计算 length-1。调整：明确全命中时重算最后一个 token 或使用有效缓存 logits 的路径；禁止零长度切片。验收：完全相同 prompt、缩短 prompt、空内容在单请求和混合 batch 中均不会挂起、下溢或重复输出。

- [ ] **CAP-01：prompt 静默截断及 slot 局部边界不足**（静态确认）。位置：`src/runtime/session/sequence.rs:64`、`:106`、`:130`。写入使用 min(capacity)，未向 API 明确报告超长；token_ids 主要检查整个分配区间，不能阻止从一个槽读到另一个槽；公开写接口缺少 slot_index 验证。调整：保留 SlotSequence，显式检查 slot、start/end 和算术溢出；容量不足返回可读错误或明确截断策略。验收：capacity-1/capacity/capacity+1、非法 slot、跨槽读取均有确定结果且不污染相邻槽。

- [ ] **CAP-02：prefill 数量没有受 max_batch_size 约束**（静态确认）。位置：`src/runtime/scheduler/scheduler.rs:78`、`:107`。decode_count 有 max_batch_size 限制，但 prefill 只受 token budget 限制。当 max_slot_size > max_num_seqs 且 prompt 很短时，本轮活动序列数/采样行数可超过 max_num_seqs。调整：在保留 decode 优先和连续布局前提下，为 prefill 计入剩余序列预算。验收：多于 batch 上限的短请求，slices/lift 行数不超设计上限、后续请求能被公平调度。

- [ ] **API-01：max_tokens/top_p 等请求参数未生效**（静态确认）。位置：`src/serving/types.rs:4`、`src/serving/server.rs:365`。字段可反序列化，但 handler 未使用 max_tokens/top_p/session_mode；model 主要用于响应回显。完成原因固定 stop。调整：实现承诺支持的字段或显式拒绝；session_mode 若仅服务器级则明确告知。验收：max_tokens=1 恰好生成一 token，长度结束返回正确 finish_reason，未支持参数不会悄悄被忽略。

- [ ] **LIFE-01：客户端取消/断流缺少可靠释放路径**（待复现风险）。位置：`src/serving/server.rs:414`、`:469`。释放在 await 循环/stream body 正常结尾，future/stream 被 drop 时可能不执行。调整：增加与请求租约绑定的取消清理，确保没有活跃 worker 再使用后才回收。验收：生成中断开 SSE 和普通 HTTP，再请求可复用槽；没有永久占槽或访问已回收状态。

- [ ] **LIFE-02：worker 无可调用的停止/回收接口**（静态确认）。位置：`src/runtime/executor/executor_pool.rs:40`。start 消费 self、线程 detached，shutdown 标志没有公开设置途径，也未保存 JoinHandle。RuntimeContext 释放时 raw pointer 使用者的生命周期需要审计。调整：保留 ExecutorPool，提供受控 shutdown+join 并确保 buffer 最后释放。验收：多次创建/关闭 runtime 不累积线程、不挂 barrier、不访问已释放序列内存。

- [ ] **CPU-01：模型和推理 worker 数实际使用 api_threads**（静态确认）。位置：`src/runtime/init.rs:106`、`:130`、`src/runtime/config.rs:62`。model.set_thread_num 和 ExecutorPool 都取 api_threads（默认 api_server_count=2），计算得到的 blocking_threads 并非推理 worker 数。调整：明确 API、Tokio blocking、推理 worker 三种数量，用同一推理数分配 scratch 并启动执行器，保留 CLI 架构。验收：请求的推理线程数与启动 worker、算子 scratch 容量一致；API 线程数变化不意外改变计算并行度；零值配置被拒绝。

- [ ] **TEST-01：处理 33 项失败而不掩盖断言**（已观察）。详见[合并报告](lift_index_merge_report.zh-CN.md)。24 项缺少资源，9 项数值/形状/参数相关。先给资源测试提供可配置 fixture；再迁移旧 total_size/lift_size/游标调用；分别在两父分支建立数值基线。验收：每项失败能归类为资源问题、测试迁移、既有缺陷或合并回归；不以放宽阈值/跳过测试替代根因说明。

## P2：保留能力后的完善

- [ ] **COMPAT-01**：为 Experts→Expert、模块移动、Operator::run 变更列出迁移表；如有外部用户，评估类型别名/兼容导出。恢复“完整模型对齐+张量 dump”的等效工具能力时，应使用新 Scheduler/SequenceSlice，不恢复旧 runtime。
- [ ] **PERF-01**：在正确性验收后比较原 develop、原 lift_index、合并版的长 prefill、纯 decode、混合 batch 和 MoE 峰值内存。特别记录 LiftVector 单线程压缩、BRGEMM/QKV 快路径、线程设置与 aligned loader 的影响。
- [ ] **DOC-01**：同步 README、CLI、运行时设计文档和生成 wiki；注明本轮保留源分支实现不代表其中所有边界行为已正确。固定性能验收工具链，保存模型/配置/CPU/版本信息。

## 建议执行顺序与关闭条件

1. 先复现 IDX-01 和 CON-01/02，建立新索引语义的单步正确性与同步约定。
2. 处理 SES-01/02、CACHE-01、CAP-01/02，保留 SlotManager 的会话复用能力。
3. 处理请求参数、取消/停止、推理线程数，再迁移测试和完整模型对齐工具。
4. 最后做性能对比与文档同步。

每项以独立 Git 提交记录问题、修复和验收结果；关闭条目时补充提交号与实际测试证据。此清单是静态审查和已有测试的汇总，不是并发安全/数值正确性的完整证明。
