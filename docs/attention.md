# CPU 上 Causal Flash Attention（含 GQA）的静态并行分配完整总结（修正版）

---

# 1️⃣ 问题背景

我们讨论在 **CPU 上实现 Causal Flash Attention（带 GQA）** 的静态并行策略，目标是：

* 静态任务分配（无动态调度）
* 最大化 CPU 利用率
* 最大化 KV cache 复用
* 不同 head 之间无需同步
* 同一时刻优先让所有核处理同一个 KV head

前提假设：任务数可以覆盖计算核数（tasks ≥ C）。

---

# 2️⃣ 参数定义

| 参数         | 含义              |
| ---------- | --------------- |
| C          | CPU 核数（固定）      |
| H          | Q head 数        |
| G          | KV group 数（GQA） |
| B          | batch size（张量维度，非切分维度） |
| L          | 序列长度            |
| causal     | 是否为自回归（下三角）     |

---

# 3️⃣ 计算结构

## 3.1 最小计算单元

```
(q_row_i × K[0..i] × group)
```

约束：

* 第 `i` 行只访问 `0..i` 的 K/V（因果下三角）
* 不做 block 切分，只按行计算

---

## 3.2 外层执行顺序

```
for batch in B:                 # 保留张量语义
    for kv_head in KV_heads:
        并行处理当前 kv_head（只沿 L 切分）
```

* 所有核先完成一个 KV head
* 然后进入下一个 KV head
* head 之间无需同步
* `batch` 维度不用于任务切分，仅作为样本索引

---


## 🎯 核心结论

> 在“任务数可覆盖核数（tasks ≥ C）”这一前提下，可以稳定实现满核并行。
> 不需要考虑 G 与 C 的关系。
> `batch size` 存在于 tensor 形状中，但不参与并行切分。

原因：

* 因果 attention 的计算量近似 O(L²)
* 可计算元素总数约为 L×(L+1)/2
* 任务总量远大于 C
* 即使每个核处理完整 group，也足够填满所有核

---

## 🔹 并行策略

* 优先固定 `kv_head` 提高 KV cache 复用
* 仅沿 `L` 维度按核数做静态切分
* 每个核负责若干连续行
* 不拆 group
* 不拆 Q head

### 线程执行结构

```
rows_per_core = ceil(L / C)
core_k 负责 [k*rows_per_core, min((k+1)*rows_per_core, L))

for assigned q_row_i:
    for k_col in 0..i:
        for group in G:
            计算 attention
```

每个核负责一部分行任务（静态均匀划分）。

---

## 4️⃣ 仅沿 L 维度的并行度估算

按行切分时，可用“行任务数”近似写成：

```
tasks_rows ≈ H × L
```

说明：

* 该估算针对“单个 batch 样本”的并行任务量
* 实现上不沿 `B` 维度切分任务，因此不将 `B` 纳入并行度公式
* 每核平均负责行数：`rows_per_core = ceil(L / C)`

工程上只需满足：

```
tasks ≥ C
```

下文默认该条件已满足。

当 `L` 不够大时，可增加可并行的 `Q head`。

---

## 🔹 关键特性

* L² 规模保证任务足够多
* 所有核必然沾满
* KV 数据共享 → cache 命中率高
* 完成一个 KV head 后进入下一个



# 8️⃣ 核心思想一句话

> 当 L 足够大时，L² 规模自然填满所有核；
> 实现上仅沿 L 维度切分，每个核负责若干行。

---

这是一种工程上简单、cache 友好、可预测、易实现的 CPU 静态并行策略。
