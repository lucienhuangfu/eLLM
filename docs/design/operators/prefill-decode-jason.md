# Jason: Qwen3-Coder CPU Prefill / Decode 优化记录

本文记录 Qwen3-Coder-30B-A3B 在 CPU AVX512-FP16 路径上的 prefill / decode 优化过程、已验证结论、当前热点和后续方向。目标是避免后续重复走已经证明收益不足的路径，并把优化判断和测试数据沉淀下来。

---

## 当前采用方案：SGLang 思路的 FP16 BRGEMM Attention

> 本节描述当前代码实际保留的方案，优先级高于后面的历史实验记录。后文同时包含
> 已回退方案，不能把“曾经尝试过”理解成“当前仍在使用”。

### 当前目标与边界

当前仅为 Attention 增加可选的长 prefill 后端：

```bash
ELLM_ATTENTION_BACKEND=brgemm
```

- 不设置该变量时，仍使用 eLLM 原生 Attention/GQA8。
- 多行 FP16 prefill 使用 BRGEMM；单行 tail/decode 使用 eLLM 原生 kernel。
- 不修改 MoE、MatMul、routing 等其他算子的执行路径。
- 不直接调用或链接 SGLang。`third_party/sglang` 只是指向本机源码的参考链接。
- 底层矩阵乘使用 LibTorch `CPUBlas::brgemm` 的 FP16/AMX-FP16 microkernel。
  tensor 生命周期、地址计算、packing、softmax、causal mask 和调度仍由 eLLM 管理。

当前数据类型：

| 数据 | 类型 | 原因 |
| --- | --- | --- |
| Q/K/V | FP16 | 保留当前模型和 KV cache 数据类型，避免额外全量转换 |
| packed K/V | FP16 VNNI layout | 直接供 FP16 BRGEMM 使用 |
| QK score | FP32 | 保留 softmax 前的累加精度 |
| softmax probability | FP16 | 直接作为 P×V 的 BRGEMM 输入 |
| running max/denominator | FP32 | 保证分块 online softmax 稳定性 |
| P×V accumulator | FP32 | 避免跨 column block 的累加误差 |
| 最终 output | FP16 | 与后续算子接口保持一致 |

### 当前执行流程

长 prefill 的每个 Q head 按以下流程运行：

```text
K/V FP16 原布局
  -> 按 KV head 完整预打包为 BRGEMM VNNI layout
  -> 同一 GQA 组的 8 个 Q heads 共享 packed K/V
  -> Q × packed K，FP32 score
  -> causal mask + 分块 online softmax
  -> FP16 probability × packed V，FP32 accumulator
  -> denominator 归一化并写回 FP16 output
```

Qwen3-Coder-30B-A3B 是 `32 Q heads / 4 KV heads / GQA ratio 8`。当前 BRGEMM
prefill 采用 `Q head × query row block` 二维任务，和 SGLang/PyTorch CPU
FlashAttention 的 `[head, MB]` 展开方式一致。5000 token、`M=512` 时共有
`32 × 10 = 320` 个任务，可以铺满 48 个逻辑线程，不再受纯 head-split 最多只有
32 个活跃线程的限制。

任务不是简单连续均分，而是按 causal block 的估算矩形工作量执行
longest-processing-time 静态分配：后段重 block 优先分散到当前累计负载最小的
线程。这样既保留每个 task 的大 M BRGEMM，也避免某个线程集中拿到三角形后段。

仍然没有采用 `4 KV heads × row parts` 的 grouped/hybrid 方案。该方案让一个 task
串行处理 8 个 Q heads，虽然能构造 48 个 task，但实测破坏 AMX 吞吐和缓存局部性，
TTFT 退化到 53.701s。这里新增的是“每个 Q head 独立拆 row block”，两者不能混同。

### Causal query-block 列裁剪

每个 query row block 现在只计算到该 block 最后一行可见的 key：

```text
key_end = min(sequence_index + row_block_end, sequence_end)
```

旧实现虽然在 softmax 中通过 `valid_cols` 排除了未来 token，但 QK BRGEMM 和 P×V
BRGEMM 仍会一直计算到完整 sequence end，causal 三角形外存在大量空算。新实现和
SGLang `num_keys = min(m + m_size, seqlen_k)` 的边界一致；block 内仍由逐行
`valid_cols` 保证精确 causal 语义。

### 完整 K/V 预打包与 GQA8 共享

当前不是每个 column block 临时 pack，也不是 8 个 Q heads 各自重复 pack：

1. 每个 Attention operator 持有共享 packed-KV cache。
2. 每个 KV head 的第一个 Q-head 线程创建完整 packed K/V。
3. 同一 KV head 下其余 7 个 Q-head 线程通过 `OnceLock` 等待并共享只读结果。
4. 4 个不同 KV heads 可以并行完成 packing。
5. cache key 包含 K/V 地址、stride、有效长度、head size 和内容指纹。
6. 每个 operator 最多保留 8 个 cache entry，避免请求形状变化导致无界增长。

代价是 10k 测试 RSS 相比每线程临时缓存增加约 824 MiB；收益是同一 KV head 的
重复 packing 从最多 8 次降为 1 次。

### 分块和 decode 策略

当前 BRGEMM block：

| sequence length | row step M | column step N |
| --- | ---: | ---: |
| `<= 256` | 32 | 64 |
| `257..=1024` | 128 | 256 |
| `1025..=4096` | 256 | 768 |
| `> 4096` | **160** | **384** |

长 prefill 的新分块按每线程 L2 工作集调优。SGLang 的推导可概括为
`MB × (1 + 1 + NB + Kv)`；eLLM 需要按实际类型展开，因为 score 和 probability
分别保留为 FP32 和 FP16：

```text
workspace_bytes
  = M × [2 × sizeof(FP32)
         + N × (sizeof(FP32 score) + sizeof(FP16 probability))
         + head_size × sizeof(FP32 accumulator)]
  = M × (8 + 6N + 4Kv)
```

本机每个物理核有 2 MiB L2，25% 约为 512 KiB，`Kv=128`。原 `512×768`
需要约 2.50 MiB/线程，单个 workspace 已超过 L2；新 `160×384` 为约
441 KiB/线程，占 L2 约 21.5%。没有强行用满25%，是为了给同一物理核上的两个
SMT sibling、活跃 Q block 和当前 packed K/V block 留出空间。

环境变量可用于重新测试，不影响未设置变量的默认路径：

```bash
ELLM_ATTENTION_BRGEMM_ROW_STEP=160
ELLM_ATTENTION_BRGEMM_COL_STEP=384
```

BRGEMM 只处理 `row_count > 1`。当 `row_count == 1` 时切回原生 Attention，并强制
`col_step=32`，原因是原生 regular kernel 的 score 临时块为 32；不能沿用长
prefill 的 768，否则会越界。decode 保持按 32 个 Q heads 并行，不强制退化成只有
4 个 KV-head tasks 的 GQA8 group fusion。

启用 `ELLM_ATTENTION_BACKEND=brgemm` 时，`head × row-block` 和 causal 列裁剪
默认同时启用。定位回归时可分别显式设置：

```bash
ELLM_ATTENTION_HEAD_ROW_SPLIT=0
ELLM_ATTENTION_CAUSAL_COL_PRUNE=0
```

### 哪些部分参考了 SGLang

参考源码：

- [`extend.cpp`](../../../third_party/sglang/sgl-kernel/csrc/cpu/extend.cpp)
- [`flash_attn.h`](../../../third_party/sglang/sgl-kernel/csrc/cpu/flash_attn.h)
- [`vec_pack.h`](../../../third_party/sglang/sgl-kernel/csrc/cpu/vec_pack.h)
- [`vec.h`](../../../third_party/sglang/sgl-kernel/csrc/cpu/vec.h)

| 部分 | 与 SGLang 的关系 |
| --- | --- |
| 长度分档和 M/N block 选择 | 采用 SGLang CPU extend-attention 的分块思路和主要取值 |
| `Q head × row block` 任务展开 | 采用 SGLang/PyTorch CPU FlashAttention 的 `[head, MB]` 并行维度 |
| query-block causal key 上界 | 采用 SGLang `min(m + m_size, seqlen_k)` 的裁剪思路 |
| K/V VNNI layout | 采用 SGLang `vec_pack.h` 使用的 BRGEMM 输入布局概念 |
| FP16 BRGEMM + FP32 accumulator | 采用其 CPU Attention 的核心矩阵计算思路 |
| 快速 FP32 exp | 多项式结构和常量参考 `vec.h` 的 `_mm512_fexp_u20_ps` |
| LibTorch low-level BRGEMM | 使用与该 CPU 路径同类的 `at::native::cpublas::brgemm` primitive |

以下部分不是从 SGLang 直接拿来调用，而是 eLLM 自己实现或保留的现有结构：

- Rust 侧的 BRGEMM 动态加载、错误检查和原生 fallback。
- Q/K/V pointer、batch/head/sequence stride 计算和静态图接入。
- causal 可见长度、跨 column block 的 online softmax 状态与最终写回。
- 基于估算工作量的 longest-first 静态线程分配、tail 处理以及 decode `col_step=32`。
- 完整 span prepack、GQA8 `OnceLock` 共享 cache、cache key 和容量限制。
- eLLM 原生 GQA8 decode/prefill kernel；默认路径没有被替换。

因此准确描述是：**参考 SGLang 的 CPU Attention 数据布局、分块和 BRGEMM
思路，在 eLLM 内重新实现一个可选后端**，而不是“把 SGLang Attention 直接接进来”。

### 当前效果和已知限制

batch=1、48 线程、16 output 的结果：

| 配置 | TTFT | Attention run+barrier |
| --- | ---: | ---: |
| 5k：共享 prepack、纯 head-split | 17.452s | 3.784955s |
| 5k：`Q head × row block` | 16.975s | 3.066401s |
| 5k：causal 裁剪和负载均衡，旧 `512×768` | 15.299s | 1.751839s |
| 5k：再加 L2 分块 `160×384` | 15.475s | **1.541649s** |
| 用户提供的 SGLang 5k PyTorch Attention self time | — | 1.466938s |
| 10k：旧共享 prepack、纯 head-split | 38.309s | 13.4336s |
| 10k：causal 裁剪和负载均衡，旧 `512×768` | 32.786s | 6.155484s |
| 10k：再加 L2 分块 `160×384`，48 线程 | **31.920s** | **5.645829s** |
| 10k：L2 分块 `160×384`，24 个物理核 | 35.602s | 5.900723s |
| 用户提供的 SGLang 10k 数据 | 37.600s | 5.467619s |

5k 相对旧 BRGEMM 路径：

- TTFT 减少 2.153s，约 12.3%。
- Attention wall time 减少约 53.7%。
- Attention barrier 从 1.875085s 降至 0.150176s，减少约 92.0%。
- 48 个线程的 Attention 累计 run 范围约 1.480s～1.678s，长尾已明显收敛。

L2 25%附近的 5k 分块筛选如下。TTFT 会受其他算子波动影响，因此选择分块时以
Attention wall time 为主：

| M×N | 每线程 workspace | TTFT | Attention run+barrier |
| --- | ---: | ---: | ---: |
| `512×768` | 约 2.50 MiB | 15.299s | 1.751839s |
| `96×768` | 约 481 KiB | 15.379s | 1.691200s |
| `128×512` | 约 449 KiB | 15.505s | 1.591302s |
| `160×384` | 约 441 KiB | 15.475s | **1.541649s** |
| `192×384` | 约 529.5 KiB | 15.927s | 1.625691s |
| `256×256` | 约 514 KiB | 15.563s | 1.831801s |

`256×256` 虽然最接近 L2 的25%，但 N 太小导致 BRGEMM 调用和边缘开销增加；
`192×384` 则出现更长 barrier。实测拐点是 `160×384`，说明缓存预算只是候选生成
规则，不能替代实际 AMX/BRGEMM 测量。

10k 文件经 tokenizer 得到 10001 tokens。最终 `160×384` 相对
`512×768`：

- TTFT 从 32.786s 降至 31.920s，减少 0.866s，约 2.6%。
- Attention run 为 4.840381s，barrier 为 0.805448s，wall time 合计
  5.645829s；相对 6.155484s 减少约 8.3%。
- 相对最早纯 head-split 的 38.309s / 13.4336s，最终 TTFT 累计减少约 16.7%，
  Attention wall time 累计减少约 58.0%。
- 用户提供的 SGLang 10k Attention self time 为 5.467619s；eLLM 的纯 run 已低于
  该值，加上线程 barrier 后仍多约 0.178s。两边 profile 口径并不完全相同。

24 线程测试设置 `ELLM_THREAD_NUM=24`、`ELLM_ALLOW_LOGICAL_THREADS=0`，亲和性为
`0,2,...,46`，确认只使用物理核。旧 `512×768` 下 24 核 Attention wall 曾比
48线程少约0.260s；缩小到 `160×384` 后，48线程 wall 为5.645829s，反而比24核的
5.900723s快约4.3%。这说明较小 workspace 已缓解 SMT sibling 的 L2压力，因此当前
不再建议 Attention 单独限制到24核，全局和 Attention 都继续使用48线程。

额外尝试过在线程 workspace 中缓存 packed-KV `Arc`，希望绕过每个 column block
的共享 map mutex。5k 结果退化为 TTFT 15.363s、Attention wall 1.928098s，相比
15.299s / 1.751839s 没有收益，已回退。不要再默认增加线程本地 HashMap。

---

## 以下为历史实验记录

> 下面早期章节中的“不使用 AMX”等目标只代表当时阶段；当前可选 BRGEMM 后端
> 已使用 AMX-FP16，应以文档顶部“当前采用方案”为准。

---

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

## 10. 第三轮线程级 profile 与 attention hybrid A/B（2026-06-23）

目标：在用户 push 后的代码基础上，确认 4000 input / 16 output / 48 线程下的 barrier 来源，并重新评估 prefill attention 是否应该改成更偏 head/kv-head 的分配。

### 10.1 新增线程级算子 profile（✅ 保留）

新增 `ELLM_PROFILE_OP_THREADS=1`，在已有 `ELLM_PROFILE_OPS=1` 或 `ELLM_PROFILE_DECODE_OPS=1` 时输出每个线程本地的算子计时：

```text
prefill_profile_thread thread_id=... kind=... count=... run=... post_barrier=...
decode_profile_thread thread_id=... kind=... count=... run=... post_barrier=...
```

这个开关默认关闭，只用于定位 load imbalance，不改变计算路径。leader 线程原有的 `prefill_profile kind=...` 汇总保持不变。

### 10.2 4000 prompt 默认 sequence split profile

命令条件：

```bash
ELLM_LOAD_THREADS=16 \
ELLM_ALLOW_LOGICAL_THREADS=1 \
ELLM_BATCH=1 \
ELLM_THREAD_NUM=48 \
ELLM_MAX_OUTPUT_TOKENS=16 \
ELLM_PROFILE_OPS=1 \
ELLM_PROFILE_OP_THREADS=1 \
ELLM_PROFILE_ATTENTION_SPLIT=1 \
ELLM_PROMPT_FILE=/tmp/qwen3_coder_prompt4000.txt \
./target/release/qwen3_coder_30b_a3b
```

输入通过 tiktoken 确认为 `长度: 4000`，输出仍有语义：

```text
Hello! It looks like you're saying hello a lot!
```

结果：

| 指标 | 时间 |
|------|------|
| load_weights | 144.631s |
| build_graph | 176.142s |
| first_token | 15.778s |
| generate(16) | 19.052s |

prefill 汇总：

| 算子 | leader run | leader barrier | per-thread run spread |
|------|------------|----------------|-----------------------|
| Attention | 4.267898s | 1.092722s | 4.267898s - 5.340467s（1.25x） |
| ExpertsMatMulSilu | 2.674073s | 1.586875s | 2.544856s - 4.226974s（1.66x） |
| MatMulAdd | 1.662207s | 0.268381s | 1.039637s - 1.887386s（1.82x） |
| ExpertsMatMulDown | 1.378105s | 0.330440s | 1.378105s - 1.495564s（1.09x） |
| MatMul3 | 2.190017s | 0.119305s | 1.595763s - 2.198816s（1.38x） |

结论：

- Attention 仍然是最大 run hotspot，但线程 spread 不算最坏，当前慢主要是每线程实算量大。
- Silu 和 MatMulAdd 的线程 spread 更明显，是后续削 barrier 的主要候选。
- Down 已比较均衡，继续微调分配的收益可能不高。

### 10.3 Attention hybrid split A/B（⚠️ 暂不设为默认）

启用：

```bash
ELLM_ATTENTION_HYBRID_SPLIT=1
```

hybrid 模式不是纯 head split，而是 `(kv_head, row_part)` 任务：

- 每个任务固定一个 KV head，内部仍处理对应的 8 个 Q heads，符合 GQA `1 KV -> 8 Q` 的结构。
- `row_part` 内仍按 causal sequence row 切分，保留三角负载平衡优势。
- 相比纯 sequence split，它让 kv-head 边界更显式，但不会让同一个 GQA 组被拆错。

A/B 结果：

| 模式 | first_token | prefill run_sum | prefill barrier_sum | Attention run | Attention barrier |
|------|-------------|-----------------|---------------------|---------------|-------------------|
| 默认 sequence | 15.778s | 12.361841s | 3.405891s | 4.267898s | 1.092722s |
| hybrid | 15.608s | 12.539037s | 3.058565s | 4.412866s | 1.028157s |

hybrid 的 first token 快约 0.17s，但 Attention run 变慢，整体 run_sum 也变慢，只是 barrier 有所下降。这个结果不足以证明 hybrid 应该成为默认策略，暂时仅作为环境变量保留，后续需要用更长输出或更多 prompt 长度重复验证。

### 10.4 下一步优先级

1. **MoE Silu 分配**：thread spread 1.66x，且 leader barrier 高。优先考虑 work-aware expert task 分配，但必须保留 expert 内 token 连续性和 gate/up pack 复用，避免重复 §9.3 中双 weight gather 退化。
2. **MatMulAdd 分配**：thread spread 1.82x。可以试 block-cyclic 或 tile-cost aware 分配，但要记住历史上从“巧妙分配”退回 naive 是为了均衡，所以只能 A/B 后保留。
3. **Attention 内部 micro-profile**：当前只知道 op 级 run，下一步需要把 QK、softmax、V accumulate、写回分段统计出来，判断是否是 softmax/score buffer 或 V accumulate 主导。
4. **decode 512 profile**：长输出测试建议固定 4000 prompt / 512 output，打开 `ELLM_PROFILE_DECODE_OPS=1` 和 `ELLM_PROFILE_OP_THREADS=1`，抓指定 step 的 decode attention/MoE spread，再决定 KV split 或 partial softmax merge 是否值得做。

### 10.5 4000 prompt / 512 output decode step profile

条件：4000 input token / 512 output token / 48 线程，仅在 `decode_step=512` 打开 decode op profile。

```bash
ELLM_LOAD_THREADS=16 \
ELLM_ALLOW_LOGICAL_THREADS=1 \
ELLM_BATCH=1 \
ELLM_THREAD_NUM=48 \
ELLM_MAX_OUTPUT_TOKENS=512 \
ELLM_PROFILE_DECODE_OPS=1 \
ELLM_PROFILE_DECODE_STEP=512 \
ELLM_PROFILE_OP_THREADS=1 \
ELLM_PROMPT_FILE=/tmp/qwen3_coder_prompt4000.txt \
./target/release/qwen3_coder_30b_a3b
```

结果：

| 指标 | 时间 |
|------|------|
| load_weights | 144.303s |
| build_graph | 176.613s |
| first_token | 15.291s |
| generate(512) | 55.143s |

输出开头仍有语义：

```text
Hello! It looks like you're saying hello a lot!

Is there something I can help you with?
```

后段出现 hello prompt 诱导下的复读/chat 片段拼接，不作为算子正确性错误证据。

decode step 512：

| 算子 | leader run | leader barrier | per-thread run spread |
|------|------------|----------------|-----------------------|
| Attention | 0.026198s | 0.000213s | 0.000007s - 0.026260s |
| ExpertsMatMulSilu | 0.014248s | 0.000491s | 0.012990s - 0.014451s |
| MatMulAdd | 0.008309s | 0.002183s | 0.000006s - 0.010441s |
| ExpertsMatMulDown | 0.007552s | 0.000431s | 0.005461s - 0.007731s |
| MatMul3 | 0.006665s | 0.000624s | 0.000027s - 0.007069s |
| MatMulTopK | 0.003818s | 0.000057s | 0.003698s - 0.003873s |

decode 与 prefill 的问题不同：

- Attention 是单步最大热点，但当前 head split 只有 32 Q heads / 4 KV heads，48 线程下必然有线程没活或活很少。
- MoE Silu 在 decode 末端反而比较均衡，说明长 decode 的首要矛盾不是 Silu routing。
- MatMulAdd/MatMul3 也有 idle 线程，属于小 batch decode 下任务粒度不足。

后续 decode 加速方向应优先考虑 **decode attention 的 KV-length split + partial softmax merge**，让同一 head 的长 KV 序列可被多个线程分担；否则仅靠 head split，48 线程无法被充分利用。这个方向需要保证 softmax 的数值正确性：每个 KV block 先算 local max/local sum/local weighted V，再按全局 max 合并 sum 和 V。

### 10.6 当前 qwen3-coder-30b-a3b benchmark 默认配置

为了让常用测试命令更短，`qwen3_coder_30b_a3b` 当前默认值调整为近期 profile 使用的配置：

| 配置 | env | 默认值 |
|------|-----|--------|
| batch size | `ELLM_BATCH` | `1` |
| max output tokens | `ELLM_MAX_OUTPUT_TOKENS` | `512` |
| runner threads | `ELLM_THREAD_NUM` | `48` |
| allow logical threads | `ELLM_ALLOW_LOGICAL_THREADS` | `true` |
| weight load threads | `ELLM_LOAD_THREADS` | `16` |

profiling 相关开关仍默认关闭：`ELLM_PROFILE_OPS`、`ELLM_PROFILE_OP_THREADS`、`ELLM_PROFILE_DECODE_OPS`、`ELLM_ATTENTION_HYBRID_SPLIT` 都需要显式设置。

### 10.7 失败的轻量分配实验（❌ 已回退）

#### MoE Silu chunked cyclic task

尝试把 `ExpertsMatMulSilu` 从逐 task cyclic：

```text
thread_id, thread_id + thread_num, ...
```

改成小块连续 task 后再 cyclic（实验 chunk=2），希望在保持一定均衡的同时增加 expert/output panel 连续性。

4000 input / 16 output / 48 线程结果：

| 配置 | first_token | Silu run | Silu barrier | Silu per-thread spread |
|------|-------------|----------|--------------|------------------------|
| 原始逐 task cyclic | 15.624s | 2.978410s | 1.250323s | 1.67x |
| chunk=2 | 16.862s | 2.698187s | 2.519629s | 1.95x |

虽然 leader 看到的 Silu run 降了一些，但 barrier 翻倍，first token 明显变差。说明当前 workload 下 Silu 的首要问题仍是尾部均衡，不能为了局部连续性牺牲 cyclic 的细粒度平衡。已回退。

#### MatMulAdd cyclic task

尝试把 `MatMulAdd` 从 contiguous range 分配改为 dense tile cyclic，希望削减 per-thread 尾部差异。

4000 input / 16 output / 48 线程结果：

| 配置 | first_token | MatMulAdd run | MatMulAdd barrier | MatMulAdd run+barrier |
|------|-------------|---------------|-------------------|-----------------------|
| contiguous range | 15.624s | 1.594411s | 0.265441s | 1.859852s |
| cyclic | 15.465s | 1.843764s | 0.107182s | 1.950946s |

cyclic 降低了 barrier，但 MatMulAdd 自身 run 变慢，run+barrier 变差；first token 的 0.16s 提升不足以证明是稳定收益。已回退。

后续分配优化要避免只看 barrier：必须同时看 `run + barrier` 和 first_token。当前更值得做的是更细的算子内部 profile，尤其是 Attention 的 QK / softmax / V accumulate 分段，以及 decode attention 的 KV split。

### 10.8 Attention kernel profile + GQA8 Q/V 优化（✅ 保留）

用户确认后把优化重心切回 Attention，gate/up 优先级后移。

新增默认关闭的 kernel 内部分段 profile：

```bash
ELLM_PROFILE_ATTENTION_KERNEL=1
```

统计位置：`qwen3_coder_30b_a3b` 在 prefill task 发送前 reset，在 first token 输出时打印：

```text
attention_kernel_profile label=first_token_prefill ...
```

4000 input / 16 output / 48 线程下，prefill attention 全部走 GQA8 kernel，regular path 为 0。优化前 kernel profile：

| 阶段 | 累计线程时间 | 占比 |
|------|--------------|------|
| GQA8 QK | 135.643291s | 60.5% |
| GQA8 softmax | 27.366722s | 12.2% |
| GQA8 value accumulate | 61.001012s | 27.2% |
| clear output | 0.189447s | 0.1% |
| **GQA8 total** | **224.200472s** | 100% |

结论：核心瓶颈是 QK。原 `dot_product_gqa8_avx512` 对每个 key row 都重复 load/convert 8 个 Q heads；但同一个 query row 内，Q 对所有 key row 是常量。

优化：针对 Qwen3-Coder 当前 `head_dim=128` 的 GQA8 path，进入一个 query row 后预先把 8 个 Q heads 的 4 个 32-wide chunk 转成 f32 lower/upper register 数组，后续每个 key row 只 load/convert K，然后复用预转换后的 Q：

```text
preload_q_gqa8_head128(q_group)
dot_product_gqa8_preloaded_q128_avx512(q_lower, q_upper, key_row)
```

非 `head_dim=128` 时仍回退原实现，保证泛化安全。

#### GQA8 Q preload

无 kernel 内部打点的正常 profile：

| 配置 | first_token | Attention run | Attention barrier | generate(16) |
|------|-------------|---------------|-------------------|--------------|
| 优化前参考 | 15.624s | 4.263051s | 1.067415s | 18.894s |
| Q preload 后 | 13.763s | 3.038789s | 0.870444s | 17.064s |

带 kernel profile 的验证（profile 会放大绝对时间，只看比例）：

| 配置 | GQA8 total | QK | softmax | value |
|------|-----------|----|---------|-------|
| 优化前 | 224.200472s | 135.643291s | 27.366722s | 61.001012s |
| Q preload 后 | 171.004552s | 77.959180s | 28.644600s | 64.263167s |

QK 累计线程时间下降约 42.5%，Attention op wall time 下降约 28.7%。输出开头仍正常：

```text
Hello! It looks like you're saying hello a lot!
```

#### GQA8 value block accumulate

Q preload 后，value accumulate 成为新的大块。原实现每处理一个 value row 都调用一次 `add_weighted_value_gqa8_avx512`，对 8 个 Q heads 的 output 反复 load/store：

```text
for value row in block:
  load output head0..7
  output += weight * value
  store output head0..7
```

优化：针对 `head_dim=128`，每个 32-wide chunk 先 load 8 个 heads 的 output 到寄存器，循环一个 32-col block 内的 value rows 做 FMA，最后每个 head 只 store 一次：

```text
for chunk in 0..4:
  acc[8] = load output heads
  for value row in block:
    value = load V chunk
    acc[head] += weight[row][head] * value
  store acc[8]
```

非 `head_dim=128` 继续走原实现。

正常 profile：

| 配置 | first_token | Attention run | Attention barrier | generate(16) |
|------|-------------|---------------|-------------------|--------------|
| Q preload | 13.763s | 3.038789s | 0.870444s | 17.064s |
| Q preload + value block | 13.279s | 2.681648s | 0.630918s | 16.673s |

kernel profile 验证（profile 会放大绝对 wall time，只看累计阶段比例）：

| 配置 | GQA8 total | QK | softmax | value |
|------|-----------|----|---------|-------|
| Q preload | 171.004552s | 77.959180s | 28.644600s | 64.263167s |
| Q preload + value block | 132.552652s | 72.151649s | 28.046437s | 32.207800s |

value accumulate 累计线程时间下降约 49.9%，Attention op wall time 继续下降约 11.8%。

#### GQA8 softmax weights head-major layout（✅ 保留）

value block 后继续看 softmax。原实现 softmax 先把每个 head 的 normalized scores 存到临时 `[32]`，再标量转置到 row-major `weights[offset][head]`：

```text
store normalized_scores[32]
for offset:
  weights[offset][head] = normalized_scores[offset]
```

改成 head-major：

```text
weights_by_head[head][32]
store normalized directly into weights_by_head[head]
```

这样 softmax 阶段少一次 32x8 的标量搬运。value block 读取从 `weights[row][head]` 改为 `weights_by_head[head][row]`。

正常 profile：

| 配置 | first_token | Attention run | Attention barrier | generate(16) |
|------|-------------|---------------|-------------------|--------------|
| Q preload + value block | 13.279s | 2.681648s | 0.630918s | 16.673s |
| + softmax head-major weights | 13.307s | 2.474897s | 0.526439s | 16.607s |

kernel profile：

| 配置 | GQA8 total | QK | softmax | value |
|------|-----------|----|---------|-------|
| Q preload + value block | 132.552652s | 72.151649s | 28.046437s | 32.207800s |
| + softmax head-major weights | 124.001518s | 71.556225s | 15.200622s | 37.098664s |

softmax 累计线程时间下降约 45.8%。value 因权重读取布局变化上升一些，但总 GQA8 累计时间仍下降约 6.5%。正常 first token 基本持平，Attention run 继续下降，保留。

下一步 attention 优先级：

1. QK 仍是最大块，可以继续看 K 侧预取/布局、减少 f32 reduce 开销，以及是否能一次处理更多 key row。
2. softmax head-major 后 value 有轻微回涨，可继续看 value block 内权重读取和寄存器调度。
3. decode 侧仍需要 KV-length split + partial softmax merge，因为 head split 无法铺满 48 线程。

## 11. SGLang 思路的 BRGEMM Attention 对照实验（2026-07-16）

### 11.1 新的对照口径

用户补充的同机 CPU SGLang TTFT 为：

| input tokens | SGLang TTFT |
| --- | ---: |
| 4000 | 14.97s |
| 8000 | 29.70s |
| 10000 | 37.60s |

本轮重点使用 10008 token、batch=1、16 output、48 线程对照。SGLang 数据只有在
prompt 模板、batch 和线程配置一致时才是严格的逐项对比；这里主要用于判断 10k
attention 路径是否已经接近其数量级。

### 11.2 实现边界

新增实验后端通过 `ELLM_ATTENTION_BACKEND=brgemm` 开启。它不是直接调用
SGLang：tensor 寻址、causal mask、online softmax、VNNI packing 和调度仍由 eLLM
实现，只动态使用 LibTorch 的底层 FP16 BRGEMM microkernel。参考源码通过
[`third_party/sglang`](../../../third_party/sglang) 查看。

数据类型保持为：

- Q/K/V、softmax probability 和最终 output：FP16。
- QK score、online softmax denominator 和 P×V accumulator：FP32。
- 单行 decode 继续走 eLLM 原生 kernel；多行 prefill 才走 BRGEMM。

长度相关 block 采用 SGLang CPU extend-attention 的思路：长上下文使用
`row_step=512`、`col_step=768`。默认后端不变，未设置环境变量时仍走 eLLM 原生
Attention/GQA8。

### 11.3 完整 K/V prepack（✅ 保留）

BRGEMM 初版按 column block 重复 pack K/V。第一次优化改为把完整 K、V span
转换为 BRGEMM VNNI layout，然后在全部 column block 中复用。

10k 结果：

| 配置 | TTFT | Attention run | Attention barrier | run+barrier |
| --- | ---: | ---: | ---: | ---: |
| BRGEMM 初版 | 40.909s | 13.5625s | 2.1743s | 15.7368s |
| 完整 K/V prepack | 39.400s | 6.4131s | 7.7852s | 14.1983s |

`run` 与 `barrier` 会因为 leader 线程工作位置变化而互相转移，因此判断时以
`run+barrier` 和 TTFT 为主。完整 prepack 的 attention wall time 下降约 9.8%，
TTFT 下降约 1.51s，保留。

### 11.4 GQA8 共享 KV prepack（✅ 保留）

完整 prepack 后仍有一个明显重复：head-split 下 8 个 Q heads 分布在不同线程，
但它们对应同一个 KV head，原先每个线程都会独立 pack 同一份 K/V。

新实现为每个 Attention operator 增加共享 cache：

- 同一 KV head 的第一个线程负责 pack。
- 其余 7 个 Q-head 线程等待同一个 `OnceLock`，随后共享只读 packed K/V。
- 4 个不同 KV heads 仍可并行 pack。
- cache key 包含 K/V 地址、stride、长度、head size 和内容指纹；每个 operator
  最多保留 8 个 entry，避免无界增长。

10k 结果：

| 配置 | TTFT | Attention run | Attention barrier | run+barrier | RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 每 Q head 独立完整 prepack | 39.400s | 6.4131s | 7.7852s | 14.1983s | 174949372 KiB |
| GQA8 共享 prepack | **38.309s** | 11.8797s | 1.5539s | **13.4336s** | 175792968 KiB |

TTFT 再下降 1.091s，Attention wall time 再下降约 5.4%。RSS 增加约 824 MiB，
符合每层保留 4 个 KV head packed K/V 的预期。当前距离用户给出的 SGLang 10k
37.6s 约 0.71s。

### 11.5 强制 grouped/hybrid BRGEMM（❌ 已回退）

尝试把长 prefill 强制改为 `4 KV heads × 12 row parts = 48 tasks`，希望占满全部
逻辑线程。实测：

```text
TTFT: 53.701s
Attention run: 26.4525s
Attention barrier: 2.3322s
```

每个 task 内串行处理 8 个 Q heads，破坏了 AMX 大 M block 的吞吐和缓存局部性；
减少 idle thread 并不等于减少 wall time。已经恢复 head-split，不应再默认强制
grouped/hybrid。

### 11.6 AVX512 寄存器转置 packing（❌ 已回退）

参考 SGLang `vec_pack.h`，独立实现过：

- K：16×16 个 32-bit（每个包含两个 FP16）寄存器转置。
- V：2×32 FP16 寄存器交织。
- 非 32 倍数 tail 继续走标量。

正确性测试通过，但 10k 实测没有收益：

| 配置 | TTFT | Attention run+barrier |
| --- | ---: | ---: |
| 共享标量 pack | **38.309s** | **13.4336s** |
| 共享 AVX512 pack | 39.111s | 13.5070s |

原因是共享后每个 KV head 只 pack 一次，packing 已不再是足够大的热点；寄存器转置、
shuffle 和 masked store 的固定成本抵消了收益。该方案已回退，不应因为“用了
AVX512”就默认认为更快。

### 11.7 FP32 accumulator 向量化写回（❌ 已回退）

尝试把最终 128 元素的 `FP32 accumulator × reciprocal(denom) -> FP16 output`
从逐元素写回改为每次 16 个元素的 AVX512 转换。正确性测试通过，但 10k 结果为：

```text
TTFT: 39.836s
Attention run: 11.5675s
Attention barrier: 2.1544s
Attention run+barrier: 13.7219s
```

没有优于共享标量 prepack 的 38.309s / 13.4336s，说明该循环不是主要瓶颈，
或者编译器已有足够好的自动向量化。已回退。

### 11.8 decode 边界修复（✅ 保留）

BRGEMM 的长 prefill block 为 768 columns，但原生 regular decode kernel 的临时
score block 是 32。取消强制 decode GQA8 后，直接沿用 768 会发生越界。

现在对 `row_count == 1` 的 BRGEMM fallback 固定使用 `col_step=32`：

- prefill 多行仍使用 512×768 BRGEMM。
- decode 单行恢复 32 Q-head 并行，不再为了 GQA8 只暴露 4 个 KV tasks。
- 10k / 16 output 完整运行通过，无 panic。

### 11.9 当前结论与下一步

从原生 eLLM 45.193s 到共享 prepack 38.309s，10k TTFT 累计下降约 15.2%；距离
37.6s 已小于 1s。下一步不应继续扩大 packed cache 或强拆 grouped task，优先级为：

1. 给 BRGEMM kernel 增加 QK、softmax、P×V、pack 四段内部 profile，确认剩余
   0.7s 是否仍在 attention，而不是 MoE 抖动。
2. 小范围 A/B `M=384/512/640` 与 `N=512/768/1024`，不要凭经验一次改大。
3. 4k、8k 分别跑同口径结果，避免只根据 10k 外推 block 最优点。
