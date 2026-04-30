# 高性能矩阵乘法原理

本文结合本仓库里的 `MatMul`、`MatMulAdd` 和 `MatMulSigmoid`，说明 CPU 上高性能矩阵乘法为什么要分块、为什么要 pack `B`、为什么要使用线程私有 scratch，以及这些设计分别解决了什么问题。

---

# 1. 计算目标和数据布局

标准矩阵乘法可以写成：

```text
C = A × B
```

其中：

- `A` 的形状是 `M × K`
- `B` 的形状是 `K × N`
- `C` 的形状是 `M × N`

在本仓库里，`B` 往往不是直接以 `K × N` 的传统布局参与计算，而是先以 `B_nt[N × K]` 的形式传入，也就是按行连续存放的 `N × K` 布局。这样做的原因是：

- 上层权重通常已经接近“按输出通道一行一行存”的格式
- kernel 更容易把 `B` pack 成 `Kc × NR`
- 避免在构造阶段反复做完整转置

这一点在 `MatMul`、`MatMulAdd` 和 `MatMulSigmoid` 中是一致的。

---

# 2. 为什么朴素乘法慢

朴素写法通常是三重循环：

```text
for i in 0..M
    for j in 0..N
        for k in 0..K
            C[i, j] += A[i, k] * B[k, j]
```

它的问题主要有四个：

1. `B[k, j]` 的访问模式对缓存不友好，容易产生大量 cache miss
2. 每次只做很少的计算，却频繁访问内存，算术强度太低
3. 没有利用 SIMD 和寄存器复用
4. 多线程时如果直接按元素或按行拆分，任务粒度太碎，调度和同步开销会变高

高性能矩阵乘法的核心，就是把这四个问题逐个解决掉。

---

# 3. 关键术语和参数

## 3.1 常用分块术语

矩阵乘法里常用以下记号：

- `MB`：`A` 和 `C` 的宏块高度
- `NB`：`B` 和 `C` 的宏块宽度
- `KC`：`K` 方向的分块长度
- `MR`：微核一次处理的行数
- `NR`：微核一次处理的列数

说明：

- `MR` / `NR` 是 GEMM 和 micro-kernel 语境里常见的记号，但不是唯一标准命名
- 不同项目可能会用不同字母组合表达同样的尺寸概念
- 在本文和本仓库里，我们把它们明确约定为“微核一次处理的行数 / 列数”

## 3.2 `MatMulParams`

本仓库把这些尺寸参数收敛到 `MatMulParams`：

```rust
pub struct MatMulParams {
    pub a_row_step_macro: usize, // MB / lda 语义复用
    pub b_row_step_macro: usize, // NB / ldc 语义复用
    pub column_step_macro: usize, // KC
    pub a_row_step_micro: usize, // MR
    pub b_row_step_micro: usize, // NR
}
```

这里的重点不是字段名本身，而是它们表达了两层信息：

- 宏块：外层怎么切
- 微块：内层微核一次算多大

## 3.3 本文中的语义约定

在本文里，可以先按下面的方式理解：

- `a_row_step_macro` 近似对应 `MB` 或 `lda`
- `b_row_step_macro` 近似对应 `NB` 或 `ldc`
- `column_step_macro` 对应 `KC`
- `a_row_step_micro` 对应 `MR`
- `b_row_step_micro` 对应 `NR`

不同 kernel 会对这些字段做不同的 stride 映射，但核心思想不变：

- 宏块决定外层分块
- 微块决定微核尺寸
- `KC` 决定一次 pack 多长的 `K` 面板

---

# 4. 单线程高性能 GEMM 的主链路

## 4.1 分块

分块，也叫 blocking / tiling，是高性能矩阵乘法最重要的原则。

把大矩阵拆成更小的块后：

- `C` 的一个小 tile 更容易放进寄存器或 L1
- `A` 的一个小块可以被复用多次
- `B` 的一个小 panel 可以被 pack 后重复喂给多个 `A` 子块

## 4.2 pack `B`

高性能 GEMM 里，`B` 往往不是直接拿原始布局去算，而是先 pack 成 `Kc × NR` 的连续面板。

原因很直接：

- 微核希望读到连续内存
- 连续访问更容易被缓存预取
- SIMD load/store 更高效
- 让 `B` 的访问顺序和计算顺序一致

在本仓库里，`MatMul` 系列会为每个线程准备一块 `KC × NR` 的 `b_panel_pool`，然后在计算前把 `B_nt` 拷贝到这块线程私有面板里。

这一步的意义是：

- 把“零散读取”变成“顺序读取”
- 把“跨列跳跃”变成“连续面板”
- 把重复访问的成本提前支付一次

这里要特别说明一点：`B_nt` 虽然本身也是按行连续存放的 row-major 数据，但它的存储布局并不等于微核所需的计算布局。

- `B_nt[N × K]` 描述的是“原始权重怎么存”
- `Kc × NR` 描述的是“微核怎么吃数据”

所以 pack 的目的不是单纯把“不连续的数据变连续”，而是把 row-major 的 `B_nt` 重排成更适合微核消费的 panel 形式。

## 4.3 寄存器阻塞与 SIMD

真正的微核通常不会一次算完整个大块，而是只算一个很小的输出 tile，比如：

- `MR = 3`
- `NR = 32`

这样做的原因是：

- `C` 的 `MR × NR` 子块可以尽量常驻寄存器
- `A` 的一个标量可以广播后反复参与 FMA
- `B` 的一个向量可以被多个 `A` 行共享

在 `src/kernel/x86_64/f16_512/matmul_block.rs` 里，`3 × 32` 就是一个典型的广播式微核：

- `A` 取 3 行
- `B_panel` 取 32 列
- 每次沿 `Kc` 做广播乘加

这类设计的本质是提高“每次从内存读进来一份数据，能做多少次计算”的比例。

现代 CPU 之所以适合做矩阵乘法，还因为它们有：

- SIMD 向量指令
- FMA（fused multiply-add）
- 多级缓存

FMA 的好处是：

- 一条指令完成乘法和加法
- 中间结果不需要单独落地
- 浮点吞吐率更高

在 FP16 路径上，`_mm512_fmadd_ph` 就是典型例子。

## 4.4 线程私有 scratch

如果多个线程共享同一块打包缓冲区，就会出现：

- 争用
- 伪共享
- 不必要的同步

所以本仓库里每个线程都有自己的 `b_panel_pool`，`MatMulSigmoid` 还额外准备了 `acc_pool`。

这带来的收益是：

- 线程之间没有写冲突
- scratch 生命周期清晰
- 访问模式稳定
- 更容易保持高吞吐

## 4.5 静态切分

这里采用的是静态任务分配，而不是动态 work-stealing。

原因是：

- 矩阵乘法的 tile 工作量是可预测的
- 静态切分没有运行时调度成本
- 对 cache 亲和性更友好

在 `MatMul::run` 中，整体 tile 会先按 `assign(total_tiles, thread_num, thread_id)` 切给线程，再由线程自己推进内部的 `K` 循环和 pack 流程。

---

# 5. 多线程分块与连续性

## 5.1 输出矩阵按 tile 切分

如果要把矩阵乘法做成多线程版本，最常见、也最稳妥的方式是按输出矩阵分块，而不是按输入元素分块。

推荐的做法是：

```text
1. 先把 C 切成 MB × NB 的输出 tile
2. 把每个 tile 视为一个独立任务
3. 用静态分配把任务区间切给线程
4. 线程内部再沿 K 方向做 KC 分块
5. 每个线程使用自己的 B_panel / accumulator scratch
```

这样做的好处是：

- 每个 tile 写回的是 `C` 的独立区域，不会发生写冲突
- 一个线程可以持续复用自己的 scratch，减少反复申请和释放
- tile 之间边界清晰，便于静态切分和负载控制
- 线程之间几乎不需要同步

## 5.2 为什么要尽量连续

如果希望多线程分块除了“独立”之外，还尽量“连续”，最重要的原则是：**让线程拿到的任务和 `C` 的 row-major 布局对齐**。

对本仓库这种按行主序存储的输出矩阵来说，比较理想的线程任务组织方式是：

- 先按 `MB × NB` 把 `C` 切成 tile
- 再把 `tiles_m × tiles_n` 展平成一维任务序列
- 让 `assign()` 分配连续的任务区间
- 尽量让同一个线程处理相邻的 `tm` / `tn` 区间

这样做的直接收益是：

- 同一线程会连续写回相邻的 `C` 区域
- L1 / L2 预取更容易命中
- 线程之间更不容易交叉写同一条 cache line
- tile 的访问模式更稳定，也更容易被 CPU 预测

在实际切分时，通常有两种偏好：

1. **优先连续 `m` 带**
   - 让一个线程尽量处理一段连续行
   - 对 `C` 的写回最友好
   - 适合输出矩阵比较宽、每行工作量较大的情况

2. **在连续 `m` 带内顺序推进 `n`**
   - 让线程按列块顺序扫过该行带
   - 保持 tile 的空间局部性
   - 适合 `NB` 已经对齐、且 `n` 方向 tile 足够整齐的情况

需要避免的是 round-robin 式分发，也就是把任务交错地派给不同线程。例如：

- `thread 0` 拿 `0, 4, 8, ...`
- `thread 1` 拿 `1, 5, 9, ...`

这种方式虽然看起来“均匀”，但会让每个线程频繁跳跃，通常不利于 cache 局部性。

所以更稳妥的策略是：

- **任务区间连续**
- **线程内遍历顺序连续**
- **写回区域连续**

如果 tile 数量本身不够多，即使切法连续，也可能喂不满线程。这时通常要从更高层补并行度，比如：

- 减小 tile 尺寸
- 增加 batch 维并行
- 在上层把不同 slice / 不同样本一起调度

## 5.3 `assign()` 的角色

本仓库里的 `assign()` 更接近“连续区间切分”而不是“按前缀和重新估权重”。

这意味着它的重点是：

- 简单
- 低开销
- 保持连续性

对于规则 GEMM，这通常已经足够好；如果任务高度不均衡，才更需要额外的 prefix 风格调度。

## 5.4 `assign()` 之后如何逐块访问

`assign(tiles, thread_num, thread_id)` 返回的是一个连续区间 `[tb, te)`，线程会按这个区间里的 tile id 依次处理任务。

在 `MatMul::run()` 里，tile id 会先还原成二维坐标：

```text
tm = t / tiles_n
tn = t % tiles_n
m0 = tm * mb
n0 = tn * nb
```

所以线程的访问顺序可以理解为：

1. 先拿到一段连续的 tile id
2. 再把 tile id 映射回 `(tm, tn)`
3. 以 `(m0, n0)` 为当前 tile 左上角
4. 在 tile 内部继续沿 `K`、`N`、`M` 方向逐层细分

更具体地说，一个 tile 的内部顺序通常是：

```text
for k0 in 0..K step KC
    pack 当前 tile 对应的 B_panel
    for nt in 0..n_blk step NR
        for mi in 0..m_blk step MR
            调微核计算一个 MR × NR 子块
```

这意味着：

- `assign()` 决定的是“哪个线程做哪些 tile”
- `tile` 的内部循环决定的是“这个线程怎么把 tile 算完”

对 row-major 的 `C` 来说，连续的 `tn` 会让线程优先访问同一 `tm` 下相邻的列块，所以写回通常也是连续的。

对 `MatMulSigmoid` 来说，这个顺序只是在中间多了一层 `acc_ptr`：

- 先用 `b_panel_ptr` 做分块累加
- 再把结果写到 `acc_ptr`
- 最后统一应用 `bias` 和 `sigmoid`

因此，从代码执行角度看，`assign()` 之后的逐块访问可以概括成：

> 线程先顺序拿连续 tile，再把 tile 还原成二维位置，然后在 tile 内按 `KC × NR × MR` 的层次继续展开。

---

# 6. 代码里的执行链路

## 6.1 `MatMul`

`MatMul` 的核心路径可以概括为：

```text
new()
  -> 预分配线程私有 b_panel_pool
  -> run()
      -> 计算 tile 范围
      -> assign() 静态切分
      -> pack B panel
      -> 调 micro-kernel
```

对应文件是：

- [`src/operators/matmul/matmul.rs`](../src/operators/matmul/matmul.rs)

这个结构的关键点有两个：

- 线程在进入 `run()` 之前就已经拿到了自己的 scratch
- 计算阶段不再做额外分配，只做 pack 和计算

## 6.2 `MatMulAdd`

`MatMulAdd` 的语义不是单纯的 `C = A × B`，而是：

```text
C = residual + A × B
```

所以它会先把 `residual` 覆盖到 `output`，再在这个基础上继续做 matmul 累加。

这样设计有两个好处：

- 最终输出只写一次到目标矩阵
- residual 和 matmul 的融合减少了额外的中间 buffer

对应文件是：

- [`src/operators/matmul/matmul_add.rs`](../src/operators/matmul/matmul_add.rs)

## 6.3 `MatMulSigmoid`

`MatMulSigmoid` 的执行链比普通 GEMM 多了一层 accumulator 管理。

它的计算语义可以理解为：

```text
acc = A × B + bias
output = sigmoid(acc)
```

这里的 `acc_pool` 就是给中间累加结果准备的线程私有缓冲区。

它的价值在于：

- `A × B` 的累加过程可以先稳定落在连续 scratch 里
- 最后再统一做 sigmoid，避免边算边改输出状态
- 对 fused kernel 来说，accumulator 结构更清晰，也更容易和不同 bias 策略组合

`acc_pool` 的尺寸按 `threads × (MB × NB)` 预分配，也就是：

- 每个线程一块独立的 accumulator 区域
- 每块大小足够覆盖一个 tile 的中间结果

对应代码在：

- [`src/operators/matmul/matmul_sigmoid.rs`](../src/operators/matmul/matmul_sigmoid.rs)

这里还要注意一点：`b_panel_pool` 和 `acc_pool` 解决的是两类不同问题。

- `b_panel_pool` 负责把 `B_nt` 打包成适合微核的连续面板
- `acc_pool` 负责保存当前 tile 的中间累加结果

前者服务于“输入访问效率”，后者服务于“中间计算状态管理”。

## 6.4 `MatMulSigmoid::run()`

`MatMulSigmoid::run()` 的大致顺序是：

```text
1. 根据 prefill_size 计算 m_run
2. 将 M 向上 pad 到 MR 的倍数
3. 计算 tiles_m / tiles_n
4. 用 assign() 把 tile 分给线程
5. 每个线程拿自己的 b_panel_ptr 和 acc_ptr
6. 逐 tile 调用 block_matmul_sigmoid::matmul_sigmoid
```

这说明 `acc_pool` 并不是全局共享的结果缓存，而是一个“线程本地、tile 可复用”的中间工作区。

## 6.5 kernel 层

真正的分块计算逻辑不在 operator 层，而在 kernel 层：

- [`src/kernel/scalar/matmul_block.rs`](../src/kernel/scalar/matmul_block.rs)
- [`src/kernel/scalar/block_matmul_sigmoid.rs`](../src/kernel/scalar/block_matmul_sigmoid.rs)
- [`src/kernel/x86_64/f16_512/matmul_block.rs`](../src/kernel/x86_64/f16_512/matmul_block.rs)

它们分别负责：

- `matmul_block`：最基础的块状 GEMM 微核
- `block_matmul_sigmoid`：在块状乘法后再接 sigmoid
- `f16_512::matmul_block`：FP16 AVX-512 路径下的 3 × 32 微核

也就是说，operator 层负责“任务组织”，kernel 层负责“具体怎么算”。

---

# 7. 还能继续优化什么

结合这篇文档，后续还值得继续优化的方向主要有这些：

## 7.1 预 pack 静态权重

如果 `B` 是模型权重，而且在推理阶段是静态的，那么最理想的做法是：

- 在 layer 初始化时就把 `B_nt` 预先 pack 成固定面板布局
- 后续每次 GEMM 直接消费 packed `B`
- 避免每个 tile 运行时重复 pack

这通常是最容易看到收益的方向之一。

## 7.2 更强的寄存器阻塞

当前已经有 `MR × NR` 的微核思路，下一步可以继续减少依赖链：

- 同时维护更多 accumulator
- 交错安排 `load A`、`load B` 和 FMA
- 让 CPU 的乱序执行有更多可并行的指令

## 7.3 让参数更贴近 cache

`MB / NB / KC / MR / NR` 不是静态真理，而是要根据真实硬件调参：

- `KC` 太大，`B_panel` 压力会上来
- `KC` 太小，pack 和循环开销会变大
- `MB`、`NB` 不合适，会让 tile 过碎或过大

## 7.4 双缓冲和 prefetch

如果 pack 和 compute 的开销都比较明显，可以考虑：

- 一个 buffer 在算
- 另一个 buffer 预取或 pack 下一块

另外，软件 prefetch 可以作为补充，但一般应优先验证实际收益，再决定是否加进热路径。

## 7.5 更高层的并行度

当 `tiles_m × tiles_n` 不够多时，可以把更高层的 batch、slice 或样本维度也纳入调度，这样才能把线程真正喂满。

---

# 8. 一张简化心智图

如果要把这套实现压缩成一张心智图，可以记成下面这个层次：

```text
MatMul / MatMulAdd / MatMulSigmoid
    -> 静态 tile 切分
    -> 每线程私有 scratch
        -> b_panel_pool: B 的连续面板
        -> acc_pool: 中间累加结果
    -> block / micro-kernel
        -> MR × NR 寄存器阻塞
        -> SIMD / FMA
    -> 写回 output
```

这张图的重点是：

- 上层解决“任务怎么切”
- 中层解决“数据怎么放”
- 底层解决“每个小块怎么快算”

如果这三层都做对了，矩阵乘法性能通常就会明显上一个台阶。
