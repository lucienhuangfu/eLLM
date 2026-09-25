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
