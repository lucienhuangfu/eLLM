# Jason: Qwen3-Coder CPU Prefill / Decode 优化记录

本文记录 Qwen3-Coder-30B-A3B 在 CPU AVX512-FP16 路径上的 prefill / decode 优化过程、已验证结论、当前热点和后续方向。目标是避免后续重复走已经证明收益不足的路径，并把优化判断和测试数据沉淀下来。

## 1. 测试背景

当前主要测试环境和约束：

* 模型：Qwen3-Coder-30B-A3B-Instruct
* batch size：1
* CPU：24 物理核 / 48 逻辑线程
* 主要运行线程：`ELLM_THREAD_NUM=48`
* 允许超线程：`ELLM_ALLOW_LOGICAL_THREADS=1`
* 权重加载线程：`ELLM_LOAD_THREADS=16`
* 当前优化目标：不使用 AMX，优先优化 AVX512-FP16 路径

常用测试命令形态：

```bash
ELLM_LOAD_THREADS=16 \
ELM_ALLOW_LOGICAL_THREADS=1 \
ELLM_BATCH=1 \
ELLM_THREAD_NUM=48 \
ELLM_MAX_OUTPUT_TOKENS=512 \
./target/release/qwen3_coder_30b_a3b
```

注意：实际命令中环境变量名应为 `ELLM_ALLOW_LOGICAL_THREADS`。

prompt token 数量已经用 tiktoken / chat template 校验过。对于当前二进制，`hello` 重复次数和最终输入 token 数大致对应为：

| 目标输入 token | `hello` 重复次数 |
| --- | --- |
| 1000 | 992 |
| 2000 | 1992 |
| 3000 | 2992 |
| 4000 | 3992 |
| 40000 | 39992 |

40000 input / 512 output 已经确认不是命令行长度问题，而是静态图 / sequence 相关内存压力导致进程在 build / prefill 前后被 OOM kill。因此后续长上下文实验应优先解决静态图和 cache 容量问题，而不是只看算子内核。

## 2. 已观测现象

### 2.1 SGLang 对比数据

用户给出的 CPU SGLang 数据如下，`generate_s` 是完整生成时间，包含 prefill：

| input tokens | first token | generate |
| --- | ---: | ---: |
| 1000 | 4.8319s | 72.9746s |
| 2000 | 9.3444s | 85.0503s |
| 3000 | 14.3349s | 96.5362s |
| 4000 | 19.1919s | 107.4819s |

本项目优化后的 1000 / 2000 / 3000 / 4000 input，512 output 的一次结果：

| input tokens | first token | generate | decode approx |
| --- | ---: | ---: | ---: |
| 1000 | 3.819s | 34.478s | 30.659s |
| 2000 | 8.421s | 43.154s | 34.733s |
| 3000 | 13.299s | 49.796s | 36.497s |
| 4000 | 18.877s | 58.541s | 39.664s |

当前项目在总 generate 上明显快于这组 SGLang 数据，但仍有两个问题：

* prefill 虽然已经接近甚至略快，但目标是明显快于 SGLang。
* decode 随 prompt 增长仍有明显斜率，长上下文下迟早会被 KV scan 成本拖慢。

### 2.2 first token 计时

二进制已经加入时间戳，用于区分：

* `load_weights`
* `build_graph`
* `first_token`
* `generate`
* `total`

测量口径已经确认：`first_token` 从 prefill 相关执行开始计入，不应混入权重加载；`build_graph` 是单独打印的阶段。

## 3. Attention 结论

### 3.1 当前 GQA 结构

Qwen3-Coder-30B-A3B 的 attention 形态：

```text
num_attention_heads = 32
num_key_value_heads = 4
attention_heads_per_kv = 8
```

也就是：

```text
kv_head 0 -> q_head 0..7
kv_head 1 -> q_head 8..15
kv_head 2 -> q_head 16..23
kv_head 3 -> q_head 24..31
```

设计目标是保持 `1 KV head -> 8 Q heads` 的 GQA 对应关系，尽量复用 K/V 读取，同时不要破坏 causal attention 的三角形工作量优势。

### 3.2 Prefill 切分

长 prefill 当前应优先使用 sequence split：

* 按 sequence 方向切分可以利用 causal attention 的三角形工作量估计。
* 每个线程拿到连续 row 区间，有利于 KV cache 连续访问。
* 在每个 row 区间内部，再按 `kv_head` 计算对应的 8 个 Q heads。

因此，sequence split 并不等于放弃 head/GQA 优化；正确结构是：

```text
thread -> triangle-balanced row range
for kv_head in 0..4:
    compute q_head group under this kv_head
```

### 3.3 Head split / GQA wave 的限制

单纯按 head split 在 48 线程机器上不够，因为 Q heads 只有 32 个，KV heads 只有 4 个。短 slice 或 decode 场景下，如果只按 `kv_head` 分组，则 task 数更少，线程覆盖更差。

已验证但不应默认开启的方向：

* `ELLM_ATTENTION_DECODE_GQA8=1`
* 4000 prompt / 16 output 的 decode step profile 中，单纯 GQA8 decode group 让 attention 从约 `0.0385s` 变成约 `0.0482s`
* 原因是只暴露 4 个 `kv_head` task，48 线程下并行度损失大于 K/V 复用收益

结论：decode 不能只做 GQA8 group fusion。要同时获得并行度和 K/V 复用，需要 KV column split 加 partial softmax merge。

## 4. MoE Routing 结论

用户提出的目标数据结构：

```text
atomic_expert_vector # [num_experts]
index_tensor         # [num_experts, batchseq]
score_tensor         # [num_experts, batchseq]
input_tensor         # [topk, batch * seq]

per-thread:
mini_expert_vector   # [num_experts]
mini_index_tensor    # [num_experts, mini_batch * topk]
```

这个方向的核心目标是：

1. 每个线程分别得到 topk。
2. 线程内分批统计 topk。
3. 根据线程内 `mini_expert_vector` 从全局 atomic expert vector 得到写入区间。
4. 将 expert id / token id 写入 `index_tensor`。

已尝试过 mini routing / parallel routing 变体，但在当前 batch=1、长 prefill 的测试下收益不明显，甚至略慢。原因：

* `ExpertsSoftmaxNorm` 当前只有约 `0.095s`，不是主热点。
* 额外 mini buffer、merge、atomic 和 cache 写入可能吞掉收益。
* 算子本身太小，优化 routing 的上限远低于 attention / MoE matmul。

当前保留状态：

* mini routing 实验已回退。
* 仍保留全局 `expert_counts`、`index_tensor`、`score_tensor`、`topk_indices` 的直接 routing。
* 后续如果重新做 mini routing，必须用更大的 batch 或更复杂 expert 分布单独验证，不能只看 4000 prompt / batch=1。

## 5. MoE MatMul 优化记录

### 5.1 Pack B / weight 是否已经提前

当前权重侧 pack 基本已经按“在 `new()` 中预处理”的思路完成：

* `MatMul`
* `MatMulAdd`
* `MatMul3`
* `MatMulTopK`
* `ExpertMatMulSilu`
* `ExpertMatMulDown`

这些算子的 B / weight panel 都是在构造阶段预 pack，运行时只读取 packed panel。

真正还会运行时重复发生的是 A/input 侧 packing，尤其是 MoE routed token：

```text
routed token id -> gather input rows -> pack A tile -> micro-kernel
```

这和固定权重不同，A 依赖当前 token / expert routing，不能简单全部提前到 `new()`。但可以在 full tile 热路径中绕过中间 `a_tile`，直接 gather token row 做 FMA。

### 5.2 Silu gather rows

`ExpertsMatMulSilu` 已加入 full 3-row gather 路径：

* 原路径：先把 routed token 输入 copy 到 `a_tile`，再读 `a_tile` 做 gate/up matmul。
* 新路径：满 3 行时直接从 input base 根据 `idx_buf` gather 三个 token row，用 AVX512-FP16 broadcast + FMA 更新 gate/up accumulator。
* 单行和 tail 仍走旧路径。

4000 input / 16 output 的一次 profile：

```text
first_token: 18.181s
ExpertsMatMulSilu run=3.140656s
ExpertsMatMulDown run=1.870797s
Attention run=4.301322s
MatMul3 run=2.639338s
MatMulAdd run=1.766640s
```

### 5.3 Down gather rows

同样思路已经扩展到 `ExpertsMatMulDown`：

* 权重仍使用 `new()` 中预 pack 的 `packed_wdown`。
* full 3-row tile 不再 pack A 到 `a_tile`。
* AVX512-FP16 里直接根据 routed token id 读取三行 `nonlin[e, token, hmid]`，broadcast 后和 packed down weight panel 做 FMA。

4000 input / 16 output 最新 profile：

```text
first_token: 16.394s
generate: 19.530s

prefill_profile total_ops=486 run_sum=13.186794s barrier_sum=3.197010s

Attention             run=4.240147s barrier=1.317041s
ExpertsMatMulSilu     run=2.849049s barrier=1.415314s
ExpertsMatMulDown     run=1.326205s barrier=0.235847s
ExpertsSoftmaxNorm    run=0.095051s barrier=0.000024s
MatMul3               run=2.713471s barrier=0.121540s
MatMulAdd             run=1.883375s barrier=0.089250s
```

语义输出开头正常：

```text
Hello! It seems like you're saying hello a lot!

Is
```

这个优化不是 batch=1 专用。它作用于 MoE full 3-row routed tile；只要某个 expert 下 routed token 能凑满 3 行就能生效。decode 单 token 场景通常凑不满，因此主要收益在长 prefill。

### 5.4 MatMulAdd residual init fusion

`MatMulAdd` 原路径是：

```text
copy residual tile -> output tile
for each reduction panel:
    output += A * packed_B
```

这会让第一个 GEMM panel 再把刚写过的 output 读回来。当前已经改为：

```text
first reduction panel:
    accumulator = residual
    accumulator += A * packed_B
    store output
remaining reduction panels:
    output += A * packed_B
```

也就是把 residual 初始化融合进第一个 reduction panel，减少一次独立 copy pass 和一轮 output 读写。

4000 input / 16 output 的一次 profile：

```text
first_token: 16.725s
generate: 19.900s

MatMulAdd run=1.744402s
MatMul3    run=2.682795s
Attention  run=4.238141s
```

对比上一轮 `MatMulAdd run=1.883375s`，该算子约下降 `7.4%`。同轮 `ExpertsMatMulSilu` 有运行抖动，因此 first token 不能只用这一轮单独判断整体收益。

`MatMulAdd` 相关 release 单测已通过：

```text
8 passed; 0 failed
```

### 5.5 已撤回的 MatMul3 zero-init 实验

`MatMul3` 也存在先清零 Q/K/V head 输出，再由 GEMM 从零累加的结构。尝试过类似 first-panel zero accumulator 的改法，但 `matmul3` release 单测出现输出为 0 和 segfault，已撤回。

结论：

* `MatMul3` 暂时不保留该优化。
* 后续如果继续做，需要先单独整理 `MatMul3` 的 full tile、partial row、RoPE / RMSNorm finalize 的写入语义。
* 不能在没有完整单测通过的情况下把该方向合入主路径。

## 6. Barrier 和 CPU 利用率

当前 profile 中仍能看到较高 barrier：

```text
prefill total barrier_sum ~= 3.2s
Attention barrier ~= 1.3s
ExpertsMatMulSilu barrier ~= 1.4s
```

这说明还有线程负载不均衡问题。已观察到的来源：

* attention 的三角形 row work 即使做了估计，也可能在尾部或 kv_head wave 上不完全均衡。
* MoE expert 分布不均导致某些线程拿到更多 routed token / output tile。
* 48 逻辑线程下，超线程共享物理核资源，负载显示可能在兄弟线程间浮动。

用户确认 48 线程在当前机器上更快，因此暂时不强制物理核绑定。但如果后续做精细 profile，应继续记录：

* 24 线程 vs 48 线程
* first token
* decode 512
* 各算子 `run` 和 `barrier`

## 7. 下一步优化方向

### 7.1 Prefill 热点

当前 4000 input / 16 output 的主要热点顺序：

1. Attention：约 `4.24s`
2. ExpertsMatMulSilu：约 `2.85s`
3. MatMul3：约 `2.71s`
4. MatMulAdd：约 `1.88s`
5. ExpertsMatMulDown：约 `1.33s`

短期优先级：

* 继续降低 `ExpertsMatMulSilu` barrier。
* 检查 `MatMul3` / `MatMulAdd` 是否还有运行时 A pack、重复 zero、过细 task 或不必要 barrier。
* 对 attention 的 GQA8 long-prefill kernel 继续看 QK / softmax / V accumulate 内部占比。

### 7.2 Decode 长上下文

decode 的核心矛盾是：

* prompt 越长，每个 decode token 都要扫描更长 KV cache。
* 单纯 GQA8 fusion 并行度不足。
* 单纯 sequence / KV split 会引入 partial softmax merge。

后续更合理的路径：

1. 给 decode attention 增加 QK、softmax、V accumulate 分段 profile。
2. 做 KV column split，每个线程处理一段 KV。
3. 每段输出 partial max、partial denom、partial weighted V。
4. 用稳定 online softmax merge 合并 partial。
5. 在 `kv_head -> 8 q_heads` 内复用同一段 K/V。

验收标准：

* 4000 prompt / 512 output 下 decode 时间下降。
* 1000 / 2000 / 3000 / 4000 prompt 下 decode 斜率下降。
* 输出开头语义正常，不出现明显错位。

## 8. 不建议重复优先尝试的路径

以下方向已经试过或当前证据不足，不应作为下一步主线：

* 单独打开 decode GQA8 group fusion：并行度不足，已测慢。
* 只优化 `ExpertsSoftmaxNorm` routing：算子太小，batch=1 长 prefill 下上限不足。
* 大规模改 mini routing：需要更大 batch / 更复杂分布再验证，否则可能因为额外 buffer 和 atomic 变慢。
* 把所有 A/input 都提前 pack 到 `new()`：A 依赖运行时 token / routing / activation，不能像固定 B/weight 一样全局预 pack。

当前最有希望的主线仍是：

```text
减少运行时 A/input 搬运
降低 MoE task/barrier 不均衡
继续拆 attention 内部 profile
为长 decode 设计 KV split + partial softmax merge
```

## 9. 第二轮 A-pack elimination + AVX-512 优化（2025-06-11 已验证）

继续对 prefill 算子做 A-pack elimination 和 SIMD 优化。每步改动后跑 profile 验证（4000 input / 16 output / 48 线程）。

### 9.1 ExpertMatMulDown 2-row gather（✅ 保留）

延续 §5.3 的 3-row gather 思路。Down 的 2-row partial tile 从 `pack_a_tile + compute1_rows` 改为直接 `compute1_gather_2rows`（AVX-512 broadcast+FMA）。

**改动文件**：
- [expert_matmul_mul.rs](src/operators/expert/expert_matmul_mul.rs)：`run()` 中 2-row 路径
- [expert.rs](src/operators/traits/expert.rs)：`ExpertsDownTrait` 新增 `compute1_gather_2rows`
- 9 个单测通过

### 9.2 MatMulAdd compute_rows 3-row unroll（✅ 保留）

`compute_rows` 和 `compute_rows_init`（f16 AVX-512）从 2-row 补齐为 3-row unroll（acc0/acc1/acc2），与 `compute_init` 保持一致。当前 workload 下实际只命中 1-2 row（tail），但代码更完整。

**改动文件**：[matmul_add.rs](src/operators/matmul/matmul_add.rs) — 8 个单测通过

### 9.3 ExpertsMatMulSilu 2-row gather（❌ 已回退）

尝试对 Silu 做 2-row gather，**wall clock 从 4.798s 退化为 5.012s（+4.5%）**。

原因：gate+up 双 weight panel 的 gather kernel 每次 kc 迭代都要从非连续地址 broadcast gather，而旧路径的 `pack_a_tile + matmul_block×2` 中 pack 后的连续 buffer 被两次 matmul_block 共享 L1 cache。直接 gather 的 cache 行为更差。

### 9.4 MatMul3 compute1_init + 冗余 zero-init 移除（✅ 保留）

**问题**：`compute_head_tile_from_packed` 在 3-row tile 计算前先做 `for head_col in 0..head_dim { *dst = 0 }`（每 head 128 次标量写入），然后 `compute1`（`matmul_update_inplace_3x32_accum`）又从 C load 再累加。zero-init 是冗余的。

**方案**：新增 `matmul_update_inplace_3x32_first` kernel（累加器从 `_mm512_set1_ph(0.0)` 开始，不从 C load），通过 trait `compute1_init` → f16 specialization dispatch。首个 reduction panel 用 `compute1_init`，后续用 `compute1`，zero-init pass 完全消除。

**改动文件**：
- [matmul_rms_complex.rs](src/kernel/x86_64/f16_512/matmul_rms_complex.rs)：新增 `matmul_update_inplace_3x32_first`
- [linear.rs](src/operators/traits/linear.rs)：`MatMulkqvTrait` 新增 `compute1_init`
- [matmul3.rs](src/operators/matmul/matmul3.rs)：`compute1_init` trait dispatch + `compute_head_tile_from_packed` 零写消除
- [matmul3.rs](src/operators/matmul/matmul3.rs)：`compute_head_from_packed` 冗余 zero-init 移除（`compute_head_gemv` 内部用 `_mm512_setzero_ph` 开始）

**MatMul3 wall clock**：Baseline 2.829s → 2.750s（-2.8%），多轮 profile 一致。

### 9.5 rotate_half_rope → AVX-512 in-place（✅ 保留）

原 `rotate_half_rope` 每次调用分配 128 元素 Vec + 标量循环。替换为 AVX-512 in-place 版本，使用 `_mm512_fmul_pch`（complex multiply）处理。

**改动文件**：
- [rope.rs](src/kernel/x86_64/f16_512/rope.rs)：新增 `rotate_half_rope_avx512`
- [matmul3.rs](src/operators/matmul/matmul3.rs)：`compute_norm_rope` f16 路径调用 AVX-512 版本
- 2 个单测通过

**prefill 影响**：`compute_norm_rope` 在 prefill fast path 中仅命中 1-row tail tile（~40 次调用），影响可忽略。对 decode 路径（逐行 GEMV）帮助更大。

### 9.6 accumulator zero-init trait dispatch（❌ 已回退）

尝试用 trait method `zero_acc` 把 MoE 的 96 元素 acc 清零从标量循环改成 AVX-512 SIMD store。profile 显示 **Silu wall clock 4.063s → 4.245s（+4.5%）**，vtable dispatch 开销大于 96 次标量写入的节省。已全部移除（包括 `zero.rs` kernel 文件）。

### 9.7 最终 Profile 汇总

多次 profile 的 wall clock 范围（run+barrier）：

| 算子 | Baseline | 优化后范围 | 趋势 |
|------|----------|-----------|------|
| Attention | 5.606s | 5.37-5.45s | ↓ 3-4%（噪声） |
| Silu | 4.798s | 4.06-4.67s | ↓ 波动大 |
| MatMul3 | 2.829s | 2.75-2.79s | ↓ 2-3%（稳定） |
| MatMulAdd | 2.010s | 1.87-2.01s | ↓ 波动大 |
| Down | 1.648s | 1.60-1.66s | ↓ 波动大 |
| **first_token** | **17.084s** | **16.04-16.68s** | **↓ 3-6%** |

### 9.8 教训

- **多 weight gather kernel（Silu gate+up）不宜替换 pack 路径**：pack 后 L1 cache 复用收益 > 跳过 pack
- **单 weight gather kernel（Down）可以替换**
- **trait dispatch 对小操作（96 元素 zero）开销过大**：vtable call > 96 次标量写入
- **per-thread `run` 波动 ±30%+**，必须用 wall clock（R+B）或 first_token 判断
- **kernel 改动需要确认热路径是否命中**：rope AVX-512 只命中 tail row，prefill 收益可忽略
