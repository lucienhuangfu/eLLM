# 线性注意力融合重构：新增 FusedGatedDeltaRule，旧算子暂时保留

## 目标

大胆重构、不考虑兼容性，但**纯增量式落地**：新增融合算子把 GatedDeltaAttention 的流水线压成单算子；旧三算子（MatMulProj / CausalConv1dSilu / RecurrentGatedDeltaRule）及其 trait、builder、注册项暂时原样保留，不删除、不修改：

```
现状：MatMulProj -> CausalConv1dSilu -> RecurrentGatedDeltaRule -> MatMulAdd(out_proj+residual)
重构：FusedGatedDeltaRule(hidden -> attention output) -> MatMulAdd(保持不变)
```

融合范围 = hidden → 四路投影(qkv/z/b/a) → 因果卷积+SiLU → qk-norm → gated delta rule 递推 → gated RMSNorm → attention output `[seq, batch, value_dim]`。`GatedDeltaAttention::forward` 直接切换到新路径；旧三步路径的 builder 保留在库中但不再被该模块调用，删除下线待融合路径验证后另行请求。

## 任务划分（核心设计）

**任务数 = batch_size × num_v_heads**：

- task_id → `batch = task_id / num_v_heads`，`h = task_id % num_v_heads`；GQA 映射 `kv_head = h * num_k_heads / num_v_heads`
- `run()` 用 `assign(batch_size * num_v_heads, thread_num, thread_id)` 划分；任务内按 `computing_slices` 中属于该 batch 的 slice 顺序、逐 token 串行执行完整流水线（递推与卷积窗口的时序依赖要求严格有序）
- GQA 时共享 kv_head 的 q/k 投影与卷积重复计算（已确认接受，换取零同步）

## conv_state 新布局（仅融合路径使用）

旧布局 `[batch, conv_dim, kernel_size-1]` 按通道组织，一个任务的 q/k/v 通道分散在三段不连续区间（旧 CausalConv1dSilu 继续使用，不动）。融合算子采用**逐任务连续段**的新布局，由 `GatedDeltaAttention::new()` 按新形状分配：

```
conv_state: [batch_size, num_v_heads, 2*head_k_dim + head_v_dim, kernel_size - 1]
            段内顺序 q(head_k_dim) | k(head_k_dim) | v(head_v_dim)
```

- 任务 (batch, h) 独占一段连续内存，运行期间**直接原地读写 conv_state，无需 scratch 拷贝、无加载/写回**
- 权重侧无需重排：`conv_weight[conv_dim, kernel_size]` 仍按原通道序，compute 用三个子区间索引（q 段 `kv_head*head_k_dim` 起、k 段 `key_dim + kv_head*head_k_dim` 起、v 段 `2*key_dim + h*head_v_dim` 起），写法沿用原 `norm_scale_qk` 的 segments 模式
- `recurrent_state [batch, num_v_heads, head_k_dim, head_v_dim]` 本就逐 head 连续，形状不动，新旧路径共用
- 旧算子不兼容新布局：`causal_conv_silu` builder 保留在库中，但不再能直接用于 GatedDeltaAttention 的 conv_state（头注释中标注）

## 每任务计算：chunk 两阶段融合为单阶段（对应 modeling_qwen3_5.py）

delta rule 采用参考实现 `torch_chunk_gated_delta_rule`（[L290-L424](file:///d:/eLLM/scripts/qwen3_5/modeling_qwen3_5.py#L290-L424)）的两阶段公式，但**不沿用它的执行结构**：参考实现是先把全部 chunk 的阶段一一次性预计算完（全序列 q/k/v/beta/decay pad + reshape + 全量 UT 求解），再进阶段二的扫描循环；融合算子把它压成**单个循环：每轮迭代 = 算一个 chunk + 做一步递归**，两阶段在同一轮内背靠背完成，chunk i 的递归结束后才现场计算 chunk i+1。不存在任何跨 chunk 的预计算 pass，也不缓冲全序列中间量（scratch 只需单 chunk 容量）。任务粒度不变：(batch, head)，slice 内 token 按 `chunk_size`（常量 64，对齐参考默认值）分块：

```
// 单循环交错：每轮迭代处理一个 chunk，chunk 计算与递归在同一轮内完成
for chunk in slice.tokens.chunks(chunk_size):
    // ---- (a) 本 chunk 计算：逐 token 子循环，结果只进 scratch ----
    for t in chunk:                                     // 仍按 token 顺序（卷积窗口时序依赖）
        1. 投影 GEMV（in_proj_qkv/z/b/a, L555-L562）:
           q/k/v/z 头切片 -> scratch chunk 缓冲；
           beta = sigmoid(b_w[h]·hidden)；g = -exp(A_log[h]) * softplus(a_w[h]·hidden + dt_bias[h])
        2. 因果 depthwise 卷积 + SiLU（L239-L279）：在本任务 conv_state 私有窗口上滚动
        3. qk-norm：q_h 做 l2norm × 1/sqrt(head_k_dim)，k_h 做 l2norm（eps = 1e-6）
    // ---- (b) 本 chunk 矩阵化（L364-L394），只用到当前 chunk 的 scratch ----
    4. cum_decay = cumsum(g)；pairwise_decay = exp(cum_i - cum_j) 掩严格上三角为 0；
       ut_system = (k_beta @ k^T) * pairwise_decay [C, C]；
       intra_chunk_attn = (q @ k^T) * pairwise_decay [C, C]（供本轮递归使用）；
       前代求解（unitriangular，L388-L394 的 forward-substitution 形态，不依赖 linalg.solve）：
       new_values = ut⁻¹ v_beta；k_cumdecay = ut⁻¹ decayed_k_beta；
       q *= exp(cum_decay)；k *= exp(cum_last - cum_decay)
    // ---- (c) 本 chunk 的一步递归（L407-L415），与 (a)(b) 同轮完成 ----
    5. v_new = new_values - k_cumdecay @ S；o = q @ S + intra_chunk_attn @ v_new；
       S <- S * exp(cum_last) + k^T @ v_new            // S = recurrent_state[batch, h]，
                                                        // 更新后的 S 直接供下一轮迭代的 chunk 使用
    6. gated RMSNorm（L207-L224）：out = w ⊙ o * rsqrt(mean(o²) + eps) * silu(z)
    7. 写出：output[cache_row, h*head_v_dim .. (h+1)*head_v_dim] = out
    // 下一轮：带着更新后的 S 现场计算下一个 chunk，scratch 整体复用
```

- 尾部不足 chunk_size 的 chunk 按实际长度计算（求解维度 = 实际长度），不物化参考实现的 padding
- (a) 内 token 子循环仍严格串行（卷积滚动窗口时序依赖）；矩阵化只发生在当前 chunk 的 [C, C] / [C, dim] 小矩阵上，吃 SIMD
- chunk 内归约（cumsum / 小矩阵乘 / 前代求解）按参考实现用 f32 累加；decode（slice 长度 = 1）时单轮迭代即单 token chunk，公式与逐 token 递推等价

### 推理专用假设（不考虑训练）

本方案只服务推理前向（prefill + 自回归 decode），无训练/反向传播约束：

- conv_state / recurrent_state 直接在缓存上原地读写，无需保留激活供 backward；GQA 下 q/k 投影重算在训练侧是显存/反传负担，推理侧只是 FLOPs，可放心接受
- UT 求解固定为 forward-substitution 形态；参考实现保留 `torch.linalg.solve_triangular` 主路径与 dynamo-export 回退分支（[L384-L394](file:///d:/eLLM/scripts/qwen3_5/modeling_qwen3_5.py#L384-L394)）只是 export 兼容需要，推理无需区分
- 无 dropout / gradient checkpointing / 训练态 mask；数值对齐目标只有推理前向（generate 输出与 HF 一致）
- decode 是自回归热路径（每生成一个 token 走一遍全部层），因此 compute 实现时应包含 `chunk_len == 1` 快分支：直接走逐 token 递推公式，跳过 [C,C] 矩阵化机制（属 compute 实现阶段，仍列范围外，但设计约束在此固定）

out_proj + residual 保持独立 MatMulAdd，不融合。

## 文件改动

### 新增

**`src/operators/attention/linear_attention/fused_gated_delta_rule.rs`**（骨架，compute 留 TODO）：
- `pub struct FusedGatedDeltaRule<T>`，`#[derive(Clone)]`，指针全部 `ConstPtr`/`MutPtr` 包装：
  - 输入：`hidden_ptr` [seq*batch, hidden_size]；`qkv_weight_ptr` [conv_dim, hidden]、`z_weight_ptr` [value_dim, hidden]、`b_weight_ptr`/`a_weight_ptr` [num_v_heads, hidden]、`conv_weight_ptr` [conv_dim, kernel_size]、`norm_weight_ptr` [head_v_dim]、`dt_bias_ptr`/`a_log_ptr` [num_v_heads]
  - 状态（原地更新）：`conv_state_ptr`（新布局）、`recurrent_state_ptr` [batch, num_v_heads, head_k_dim, head_v_dim]
  - 输出：`output_ptr` [seq*batch, value_dim]
  - 维度：hidden_size、key_dim、value_dim、head_k_dim、head_v_dim、num_k_heads、num_v_heads、kernel_size、chunk_size（常量 64）、sequence_length、batch_size、m_max、k_max
  - scratch：仿 [scratch.rs](file:///d:/eLLM/src/operators/attention/full_attention/scratch.rs) 的 `Box<[T]>` 线程私有池（构造时按 thread_num 预分配，run 中零分配），每线程按 chunk 尺寸估算（C=64、head_dim=128 时 f32 约 106 KB/线程，常驻 L2）：
    - chunk 数据缓冲：q/k [C, head_k_dim]、v/z/new_values/k_cumdecay/o [C, head_v_dim]、beta/g/cum_decay [C]
    - chunk 内矩阵：两块 [C, C]（一块先存 pairwise_decay 后就地变 ut_system，另一块存 intra_chunk_attn）
    - 卷积窗口不在 scratch（直接住 conv_state 逐任务私有段）
- `run(&self, _total_size, computing_slices, thread_num, thread_id)`：`assign(batch_size * num_v_heads, ...)` 划分任务后调用 `compute`
- 双语头注释：融合范围、任务划分、单循环交错（每轮 = 一个 chunk 计算 + 一步递归，无全序列预计算 pass）、conv_state 新布局、Python 参考行号；`new()` 带 `debug_assert`（`num_v_heads % num_k_heads == 0`、`key_dim == num_k_heads * head_k_dim`、`value_dim == num_v_heads * head_v_dim`、`chunk_size >= 1`）
- `#[cfg(test)]` 最小单测：小维度构造（hidden=8、num_k_heads=1、num_v_heads=2、head_k_dim=2、head_v_dim=2、kernel=4、seq=4、batch=1 → 任务数 2；另加一组 seq > chunk_size 的用例覆盖多 chunk 分块），循环 thread_id 跑满线程，compute 为空期间断言不 panic、state/output 不被写脏

**trait**（追加到 [traits/linear_attention.rs](file:///d:/eLLM/src/operators/traits/linear_attention.rs)，现有 `RecurrentGatedDeltaRuleTrait` 保留）：

```rust
pub trait FusedGatedDeltaRuleTrait<T> {
    /// 单 (batch, head) 任务：slice 内按 chunk_size 分块单循环交错，每轮迭代 =
    /// (a) 逐 token：投影 -> conv+SiLU -> qk-norm 进 scratch
    /// (b) 本 chunk 矩阵化：cum_decay / ut_system / intra_chunk_attn / 前代求解
    /// (c) 本 chunk 一步递归：v_new / o / S 更新 -> gated RMSNorm -> 写输出；
    /// 更新后的 S 供下一轮迭代，无跨 chunk 预计算 pass
    fn compute(
        &self,
        batch: usize,
        head: usize, // v head; q/k head = head * num_k_heads / num_v_heads
        slices: &[SequenceSlice],
        scratch: &mut FusedScratchSlice<'_, T>,
    );
}
```

`mod.rs` 的 `traits` re-export 追加 `FusedGatedDeltaRuleTrait`。

### 保留不动（不删除、不修改）

- 算子文件：`src/operators/matmul/matmul_proj.rs`、`src/operators/conv/causal_conv1d_silu.rs`、`src/operators/attention/linear_attention/recurrent_gated_delta_rule.rs`（含各自单测）
- trait：`CausalConvTrait`（traits/conv.rs）、`MatMulProjTrait`（traits/linear.rs）、`RecurrentGatedDeltaRuleTrait`
- [assign.rs](file:///d:/eLLM/src/operators/assign.rs) 的 `assign_slice_channel_tile`（旧两算子仍在用）
- [operator.rs](file:///d:/eLLM/src/operators/operator.rs) 中旧三个 enum 变体及其 run / kind 臂
- [src/tensor/linear_attention.rs](file:///d:/eLLM/src/tensor/linear_attention.rs) 中 `matmul_proj` / `causal_conv_silu` / `recurrent_gated_delta_rule` 三个 builder

### 纯增量注册

- [operators/mod.rs](file:///d:/eLLM/src/operators/mod.rs)：`attention::linear_attention` 模块内声明 `fused_gated_delta_rule` 并 re-export `FusedGatedDeltaRule`；`traits` re-export 追加 `FusedGatedDeltaRuleTrait`
- [operator.rs](file:///d:/eLLM/src/operators/operator.rs) 三处新增：import、enum 变体 `FusedGatedDeltaRule(FusedGatedDeltaRule<T>)`（按字母序插入）、`run()` match 臂（`operator.run(total_size, computing_slices, cpu_num, thread_id)`，与原 RecurrentGatedDeltaRule 臂同形）、`kind()` 返回 `"FusedGatedDeltaRule"`

### 修改

**[src/tensor/linear_attention.rs](file:///d:/eLLM/src/tensor/linear_attention.rs)**：在现有三个 builder 之后新增 `fused_gated_delta_rule(...)` builder：分配 `output_tensor [sequence_length, batch_size, value_dim]`（scope 名沿用 `model.layers.N.linear_attn.*` 约定，满足 mem_pool REGEX_SET），构造并 enqueue `Operator::FusedGatedDeltaRule`；文件头注释补充新旧两条流水线并存的说明。

**[src/transformer/gated_delta_attention.rs](file:///d:/eLLM/src/transformer/gated_delta_attention.rs)**：
- `new()`：`conv_state` 改按新布局分配 `Tensor::zeros(vec![batch_size, num_v_heads, 2*head_k_dim + head_v_dim, conv_kernel_size - 1], ...)`；`norm_weight` 去掉 `#[allow(dead_code)]`，改为传入融合算子；其余权重张量不变（HF 键名不动）
- `forward()`：三步调用换成一次 `hidden_states.fused_gated_delta_rule(...)` 得到 `attn_output [seq, batch, value_dim]`，后续 view / lift_vector / matmul_add(out_proj+residual) 逻辑保持不变
- 模块头注释同步改写为两级流水线（FusedGatedDeltaRule -> MatMulAdd）描述，并注明旧三算子路径保留在库中未删除
- `#[cfg(test)] test_gated_delta_attention` 适配：断言队列中算子数量为 2（FusedGatedDeltaRule + MatMulAdd）且第一个 `kind() == "FusedGatedDeltaRule"`

## 已知权衡（写入算子头注释）

- hidden 行被每个 head 任务重复读取并重复 GEMV；GQA 时 q/k 段投影与卷积再重复 `num_v_heads/num_k_heads` 次——以计算换零中间缓冲、零同步
- 并行度上限 = batch_size × num_v_heads，chunk 间严格串行；但每轮迭代内的矩阵化（[C,C] UT 前代、小 GEMM）让 prefill 的主体计算吃上 SIMD，不再是纯标量逐 token 递推
- 代价：scratch 从每线程几百字节膨胀到 ~106 KB（f32、C=64、head_dim=128，含两块 [C,C]），超出 L1、常驻 L2，但只需单 chunk 容量、不随序列长度增长；(a) 必须缓冲完当前 chunk 的 q/k/v/z 才能进 (b)(c)
- decode（单 token chunk）是热路径：靠 `chunk_len == 1` 快分支退化为逐 token 递推公式，避开 [C,C] 机制与 scratch 往返（见推理专用假设）

## 验证

```bash
cargo check --all-targets        # Windows 链接受 brgemm dlopen 限制，用 check 验证编译层
cargo test fused_gated_delta_rule
cargo test gated_delta_attention
```

## 范围外（按需另行请求）

compute 逻辑实现、kernel 抽取到 crate::kernel、alignment 对齐脚手架（旧三算子本无对齐目录）、旧三算子的删除下线（待融合路径验证后另行请求）。