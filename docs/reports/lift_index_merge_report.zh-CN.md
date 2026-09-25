# lift_index 合并审查报告

本次将本地 `lift_index` 合并到 `develop`，采用真实的双父 merge commit，保留两条分支的提交历史。架构和索引语义以 lift_index 为主，保留可适配的 develop CPU 优化；没有使用整树覆盖或 squash。

## Git 范围与追溯

- 合并前 develop：`eb41be2b471949144de2c6fb6c28de221c715324`。
- 合入 lift_index：`4301f1fb1a71e9e579b901b718b6ac0568706887`。
- 公共祖先：`85086a0889e14b8ce448ae8fecfc70ba3f2156e2`。
- 起始工作区干净。本地 lift_index 与已有 origin/lift_index 引用相同；本次未 fetch，未核实服务器是否还有新提交，也未 push。
- 执行 `git merge --no-ff --no-commit lift_index`，逐项处理 27 个冲突路径，再将适配修复和本报告一并提交。
- 可以用 `git log --graph --oneline --decorate` 查看分叉和合流；对 merge commit 使用 `git show --format=fuller --no-patch <merge>` 查看双父，使用 `git diff <merge>^1 <merge>` 查看对 develop 的完整改动。
- 如需整体撤销，应审核后使用 `git revert -m 1 <merge>` 新建反向提交，保留历史；本次没有执行撤销。

## 合并取舍

| 范围 | 采用方式 | 影响 |
| --- | --- | --- |
| runtime / serving / CLI | 以 lift_index 的 Scheduler、ExecutorPool、SlotManager、SlotSequence、server/parser 为主 | 旧 Runner、scheduling、serving handler/config/scheduler 路径删除；外部 Rust 调用方需要迁移 |
| 切片与采样 | 采用 next_sequence_index、token_start_index、lift_index、total_size、lift_size | KV 位置、计算行位置、采样行位置分开，不能混用 |
| Attention / QKV | 保留 develop 的 BRGEMM、KV stride、head/row 分工及打包路径，适配新切片字段 | 保留性能实现，但需真实长上下文模型验证正确性和吞吐 |
| MoE / GEMM | 保留 develop 紧凑 routing 缓存、gather 路径和 Expert 单数命名，适配 lift/total 行数 | 避免重构时回退已有内存优化 |
| 权重加载 | 合入新 loader 路径，并保留 aligned f16 并行加载；initialize_runtime 使用 aligned ownership 初始化 | 其他示例 bin 仍采用 lift_index 的 Vec 并行加载路径，不保证相同加载峰值内存 |
| 许可证和发布材料 | 保留 develop 的 AGPL-3.0-only、LICENSE、README、benchmark 等无冲突更新 | 本次不改变项目许可证 |
| 冲突文档 | CLI、optimization、OpenAI 服务文档采用 lift_index 版本 | 与保留的发布文档可能存在配置说明差异，应以当前 CLI 和代码为准 |
| 对齐工具 | 两个冲突的 tokenizer alignment 程序采用 lift_index 版本 | 原先完整模型对齐/张量 dump 能力有缩减；不是等价迁移，旧实现仍可从合并第一父恢复查看 |
| 分支附带文件 | 保留 .qoder wiki、test_request.json 等分支内容 | wiki 约 3.3 MB，可能含重构前模块名称，不应视为权威 API 文档 |

## 合并中修正的问题

1. **计算行与采样行分离。** 普通 MatMul、MatMulSigmoid、ExpertSoftmaxNorm 根据 decode_only_flag 选择 total_size / lift_size；AddZip、AddRMSZip、SiLU 使用 total_size；最终 RMS 和 LM-head 使用采样行。prefill_size 与 decode_size 同时非零时，不再只处理其中一个段。
2. **原地 LiftVector 跨线程覆盖。** 输出前移时，另一线程可能覆盖尚未读取的源行。改为 thread 0 按调度器切片顺序处理，保留算子间 barrier；新增反向 worker 调用顺序回归。依赖调度器按 token_start_index 排序、lift_index 不大于源行的布局约定。该压缩步骤串行化，尚未测量吞吐影响。
3. **优化路径的索引适配。** Attention 保留优化实现并统一 next_sequence_index；MoE task metadata 统一 sequence_length。QKV 的 row-tiled 快路径仅在有真实切片时启用，空切片测试/兼容调用继续使用其原有固定 RoPE 位置语义，避免读取未提供的位置表。
4. **线程数下溢。** blocking_threads 使用 saturating_sub 后至少为 1；API 线程数超过可用数时不再无符号下溢。
5. **测试调用迁移。** 补齐新 Operator::run 参数；修正把 total_size 填成 0 或 batch+1、实际只分配 batch 行的旧调用。早期验证发生过 SIGSEGV/SIGABRT；修正后算子测试集可完整结束。MatMul、AddZip、AddRMSZip 测试还覆盖了 prefill/decode 同轮计算的行数情况。

## 验证

使用本机 native CPU 指令集、nightly Rust、离线 Cargo 依赖；库测试使用 `--test-threads=1`，避免全局内存池测试并行干扰。

| 命令 | 结果 |
| --- | --- |
| `cargo +nightly check --all-targets --offline` | 通过，包含库、bin 和测试目标的编译；有现存 warning |
| `cargo +nightly test --lib --offline runtime:: -- --test-threads=1` | 70 通过，3 失败，2 ignored；失败是 chat_template 测试缺少文件 |
| `cargo +nightly test --lib --offline serving:: -- --test-threads=1` | 27 通过，15 失败；失败均缺少 models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json |
| `cargo +nightly test --lib --offline operators:: -- --test-threads=1` | 93 通过，6 失败；见下文 |

最终全库测试 `cargo +nightly test --lib --offline -- --test-threads=1` 已完整结束：**289 通过、33 失败、7 ignored**，没有再次崩溃。按用户要求优先完成合并，不以全绿作为提交前提。

33 项失败中，24 项报模型/tokenizer/模板文件不存在（包含 2 项 Transformer attention 测试）；其余 9 项为数值、形状或调度参数相关断言：除下述 2 项算子失败，还包括 RMSNorm 内核精度断言，以及 Tensor API 的 Expert down、Expert merge、MatMul、MatMulAdd、两项 local top-k 测试。Tensor 测试可能仍传入旧的行数参数；需进一步区分测试迁移与实现问题，当前不作已解决结论。

算子失败清单：

- `operators::elementwise::complex_zip::test::test_complexmul2`：输出为零，与期望不符。
- `operators::matmul::matmul3::tests::test_matmul3_qkv_f32_72_rows`：Q 输出数值不一致。
- `decode_chain_runs_from_current_token_and_writes_next_slot`、`prefill_chain_runs_lookup_kqv_attention_and_topk_writeback`、`prefill_kqv_multithread_matches_single_thread`、`prefill_then_decode_reuses_kv_cache_and_advances_cursor`：初始化读取名为 gpt2 的本地 tokenizer 文件失败。

没有修改上述失败断言、降低精度阈值或将失败测试标为 ignored。未在两条父分支分别运行完整基线，因此不能宣称所有剩余数值失败都是历史问题。

## 需要重点注意

- **还不能把此次合并视为生产验证通过。** 缺少真实权重/tokenizer 的端到端验证；应补齐模型资源，跑 prefill→decode、混合 batch、分块 prefill、会话重用及长上下文的 Hugging Face 对齐，再做吞吐和峰值内存比较。
- **旧对齐测试本身仍需迁移。** 除缺少 gpt2 文件，一些测试仍把 prefill 最后一块的采样数量当作 decode_size，且 Scheduler 构造参数含旧语义。补齐文件后也不能假设自动通过，需按 task.lift_size/total_size 更新测试设计。
- **配置行为改变。** 新 determine_thread_config 使用 generation config 和 API 线程数，并通过 CPU 列表隔项计数估计物理核；旧 ELLM_THREAD_NUM 行为未原样保留。这不是可靠的 SMT/NUMA 拓扑识别，现有部署需核对实际 worker 数和绑核策略。
- **公开 API 不兼容。** Operator::run 新增行数参数，SequenceSlice 字段变化，加载器/运行时模块移动，Session/Serving API 重构，旧调用代码需要重新编译和迁移。
- **会话槽边界另需审计。** SlotSequence 的公开原始指针写入接口依赖调用方传入有效 slot_index/start_pos；token_ids 的检查侧重整个分配区间。此次没有扩展成完整输入边界/并发安全审计。
- **文档和示例不能替代实测。** 旧 README 性能数据不能直接代表新运行时；被替换的对齐工具和删除的旧集成测试应安排替代覆盖。
- **工具链可复现性。** rust-toolchain 仍为浮动 nightly，并新增 rust-analyzer 组件；本次 Rustup 自动同步了 nightly。推荐在正式性能验收时固定版本。生成 wiki 保留了上游 Markdown 尾空格，整仓 diff --check 可能因此报告样式问题。

## 后续保留核对与执行清单

已补充[详细调整清单](lift_index_followup_checklist.zh-CN.md)，核对 lift_index 保留范围，并逐项列出索引、并发、会话回收、容量、请求参数、线程数量和测试迁移问题的优先级、代码位置及验收条件。以该清单跟踪后续修复，本次不要求全部修复后才完成合并。


## develop 原设计的删除、替换与能力损失台账

本节补充此前概述未逐项列出的内容。比较基准是 **合并前 develop `eb41be2` → 合并提交 `d3c1c11`**，不是公共祖先；因此反映的是本次合并相对于用户原工作分支的实际变化。采用 lift_index 不表示 develop 原有能力都有等价替代。

| 编号 | develop 原设计与原位置 | 合并后的处理 | 状态、影响和后续要求 |
| --- | --- | --- | --- |
| D01 | `runtime/runner.rs` 的 ServingRunner / Runner，订阅 Tokio broadcast 的 ScheduleTask，JoinSet 管理 worker | `runtime/executor/executor_pool.rs` 的常驻线程 + Scheduler 共享任务 | **架构替换**。旧 broadcast/JoinSet 路径没有保留；应在 ExecutorPool 内补齐任务发布和 shutdown/join，不恢复旧 runner |
| D02 | `runtime/scheduling/token_counter.rs`：token 阈值或 timeout 触发，schedule_gate 互斥，task_in_flight CAS 门控，task id | ExecutorPool 主线程直接调度，其他线程轮询 has_work | **机制移除，非等价搬迁**。旧 threshold/timeout 和 in-flight 门控不再存在；需用新任务代次/发布协议保证等价的任务互斥与唤醒正确性（CON-02） |
| D03 | `runtime/scheduling/scheduler.rs` 的 BatchScheduler、SliceScheduler、每线程 prefill_list、DecodeList | Scheduler 两遍遍历，统一 slices，prefill/decode 连续 token 布局 | **架构替换**。旧 per-thread 预切片和独立 DecodeList API 被删除；保留 lift_index 的统一切片方案 |
| D04 | `runtime/scheduling/types.rs` 的 SequenceState：sequence_index、kv_index、filling_length；task 的 thread_count | SlotState 的 next_sequence_index、prompt_length、sequence_length；执行器固定 worker 数 | **状态模型/API 替换**。原有读位置/写位置的分工不能仅改字段名；需关闭 IDX-01。旧每 task 活跃线程数量接口未等价保留 |
| D05 | `runtime/batch_sequence.rs` 的 BatchSequence、batch_temperature、row_size/col_size | SlotSequence、slot_temperature、slot_count/slot_capacity | **替换**。序列/tokenizer/模板能力仍在；增加会话槽语义，外部旧类型与字段调用不兼容 |
| D06 | `runtime/scheduling/sequence_slice.rs` 的 DecodeList / DecodeLookupResult、lookup_global_index、walk_global_range | SequenceSlice + token_start_index / lift_index，LookupRMSMap 自行定位切片 | **旧抽象/API 删除，功能路径改写**。不能把旧 global index 接口当成仍然可用 |
| D07 | `runtime/runner.rs` 的 ProfileRow、算子执行/pre/post barrier 统计，ELLM_PROFILE_OPS、ELLM_PROFILE_DECODE_OPS、ELLM_PROFILE_DECODE_ALL、ELLM_PROFILE_DECODE_STEP、ELLM_PROFILE_OP_THREADS | 新 ExecutorPool 未见等效开关与输出 | **能力未迁移**。旧 profiling 环境变量不再产生对应统计；应在新执行器内恢复观测能力，再比较性能 |
| D08 | `serving/mod.rs` 的 ApiState，Semaphore + free_slots 队列；`chat_handlers.rs` 在无槽时 await permit | SlotManager acquire_session 槽满返回 SlotUnavailable，API 返回错误 | **背压行为改变**。从等待空槽转为容量不足报错；客户端需退避重试，或在新 SlotManager 前实现有界等待策略 |
| D09 | `serving/chat_handlers.rs` 读取 request.max_tokens（默认 100，至少 1），同步/流式路径参与生成停止 | 新 request 保留字段，但 server handler 不使用 max_tokens | **旧功能未保留**。这是相对 develop 的明确行为退化，非单纯 API 移动；需要在新槽/调度体系内恢复长度限制（API-01）。旧实现也需独立验证，不能直接照搬计数方式 |
| D10 | `runtime/runner.rs` 在一轮结束后对比 SequenceSnapshot，token 或 phase 改变时通知请求；旧 handler 记录 generation_starts 并处理 EOS 输出范围 | 新 TopKSoftmax 通知 EOS/容量结束，常规 token 按 write_sequence_index % 10 通知；按 prompt_length 解码 | **通知节奏和输出边界改变**。可能改变首 token/流式延迟；需验收不足 10 token、EOS 是否包含、容量终止，不保证与原 develop 等价 |
| D11 | `serving/config.rs` 的 ELLM_BATCH / ELLM_SEQUENCE_LENGTH / ELLM_CHUNK_SIZE / ELLM_SCHEDULE_TIMEOUT_MS；`model_setup.rs` 的 ELLM_THREAD_NUM、worker_threads/async_threads | 统一 CLI/ResolvedConfig + runtime/config.rs 的 api_threads/blocking_threads | **主服务配置入口替换**。旧环境变量在主服务未等价保留；个别示例 bin 仍读取部分同名变量，不能据此认定主服务兼容。需发布迁移表并处理 CPU-01 |
| D12 | `serving/model_setup.rs` 对空 eos_token_id_list 使用模型配置回退（filter 非空） | `runtime/config.rs` 仅对 None 回退，Some([]) 保持空 | **边界行为退化**。生成配置显式给空列表时，可能不识别模型 EOS，只在容量处结束；需增加空列表回退/明确校验和测试 |
| D13 | `serving/resources.rs`、`model.rs`、`model_setup.rs`、`scheduler.rs` 分层初始化与 ServingResources | runtime/init.rs / config.rs / RuntimeContext，server.rs 初始化入口 | **组织/API 替换**。模型构图、采样配置、权重加载仍存在，但不保证所有默认值/生命周期等价；aligned 加载已接回 initialize_runtime |
| D14 | 最终 norm 后调用 norm_state.lift_vector()（transformer/model.rs） | lift_index 在最后一层 attention 路径提前压缩，后续使用 lift_size；最终 norm 后不再重复 lift | **计算图设计替换**，并非丢失 lift 功能。必须保留新压缩位置与最后一层行数约定，验证 dense/MoE、混合 prefill/decode |
| D15 | `alignment/tokenizer/multi_batch_alignment.rs`、`qwen3_one_token_alignment.rs` 的模型执行、逐层 tensor dump、token 对齐输出 | 采用 lift_index 的精简 tokenizer/template 程序 | **能力缩减，尚无等价替代**。旧调试/对齐流程不能继续照旧使用；应基于新 runtime 迁移完整对齐工具（COMPAT-01） |
| D16 | `tests/qwen3_06b_integration_test.rs` | 文件删除；新增 runtime/serving 测试 | **测试覆盖移除**。新 FakeEcho/调度测试不等价于真实 Qwen 模型集成验证，需恢复等效端到端覆盖 |
| D17 | `runtime/io/*`、FromSafetensors；`runtime/spin_barrier.rs` | loader/* 整合转换函数，executor/sync.rs 提供 barrier | **移动/整合或替换，不能一概算能力删除**。直接 aligned f16 转换、并行加载已保留；旧 import 路径仍不兼容 |
| D18 | `docs/serving/parallelism_scaling.md`、两份 Llama-2 示例 config；旧运行时设计文档 | 并行扩展文档和示例 config 删除，overview/schedule 改写，新增 executor/session_management 文档 | **文档/样例移除与替换**。删去 config 不等于已证明模型支持被删除；需要按实际 loader/model 验收支持范围 |

### develop 已保留的内容，不应误记为删除

BRGEMM attention、KV stride 与 head/row 分工、QKV 打包快路径、MoE 紧凑 routing/gather 优化、内存对齐 ownership 和主初始化路径的并行 aligned f16 加载已保留并适配。Expert 单数命名、AGPL 许可证、README/benchmark 发布材料也保留。保留实现不等于其组合已经完成真实模型数值与性能验收。

### 应在 lift_index 架构内补回的 develop 能力

- [ ] 恢复 max_tokens 限制及正确结束原因，不恢复旧 ApiState/TokenCounter。
- [ ] 迁移算子和 barrier profiling 开关到 ExecutorPool。
- [ ] 给任务发布、单轮互斥、取消和 shutdown 提供明确协议，替代旧门控，不直接拷回旧调度器。
- [ ] 恢复空 EOS 列表回退或显式配置校验。
- [ ] 迁移完整模型对齐/tensor dump，并建立真实 Qwen 集成覆盖。
- [ ] 明确过载等待/拒绝策略、流式通知节奏、旧环境变量迁移及示例配置的保留范围。

### Git 追溯与删除路径原始清单

查看被删除实现使用 `git show eb41be2:<原路径>`，不需要切分支或覆盖现有文件。例如：

```bash
git show eb41be2:src/runtime/runner.rs
git show eb41be2:src/serving/chat_handlers.rs
git diff eb41be2 d3c1c11 -- src/transformer/model.rs
git diff --name-status --find-renames eb41be2 d3c1c11
```

下列是 Git 按默认重命名检测列出的 D 路径；“D”仅表示原路径消失，是否有替代以台账为准。修改但路径仍在的 alignment 程序不会出现在此列表，因此不能只看删除文件判断能力损失。

```text
docs/serving/parallelism_scaling.md
models/Llama-2-70b-hf/config.json
models/Llama-2-7b-hf/config.json
src/runtime/batch_sequence.rs
src/runtime/io/from_safetensors.rs
src/runtime/io/mod.rs
src/runtime/runner.rs
src/runtime/scheduling/initialization.rs
src/runtime/scheduling/mod.rs
src/runtime/scheduling/scheduler.rs
src/runtime/scheduling/sequence_slice.rs
src/runtime/scheduling/slice_scheduler.rs
src/runtime/scheduling/token_counter.rs
src/runtime/scheduling/types.rs
src/runtime/spin_barrier.rs
src/serving/chat_handlers.rs
src/serving/config.rs
src/serving/model.rs
src/serving/model_setup.rs
src/serving/resources.rs
src/serving/scheduler.rs
tests/qwen3_06b_integration_test.rs
```
