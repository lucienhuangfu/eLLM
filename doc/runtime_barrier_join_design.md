# eLLM Runtime Barrier / Join Design

## 1. 背景

当前 eLLM 的执行模型更接近 persistent workers + operator queue：

- 启动固定数量的工作线程。
- 每个线程都会遍历同一份 operator queue。
- 每个 operator 执行结束后，所有线程通过 barrier 同步。
- 同步完成后，再进入下一个 operator。

当前 barrier 实现在 [src/runtime/barrier.rs](../src/runtime/barrier.rs) 中，属于典型的 centralized spin barrier：

- 所有线程对同一个 `arrived` 原子计数做 `fetch_add`。
- 最后一个到达的线程推进 `generation`。
- 其他线程持续轮询 `generation` 直到变化。

这个模型在语义上是成立的，但在 CPU 热路径上，barrier 的使用层级和频率需要重新设计。

## 2. 现状问题

### 2.1 barrier 放在了 runtime 外层，而不是 operator 内部

当前模型是：

1. 所有线程一起执行当前 operator。
2. 当前 operator 执行完成后，所有线程 barrier。
3. barrier 完成后，进入下一个 operator。

这意味着 barrier 是 operator 间的 phase synchronization。

这和常见的 `parallel_for` 模型不同。`parallel_for` 的同步通常封装在单个 operator 内部：

1. 进入某个 operator。
2. operator 内部完成任务切分、线程执行、隐式 join。
3. operator 返回。
4. 调用者继续执行下一个 operator。

两者都需要“等待所有线程完成”，但同步发生的层级不同：

- 当前 eLLM：operator 外显式 barrier。
- `parallel_for`：operator 内隐式 join。

### 2.2 barrier 次数与 operator 数量、序列长度直接相乘

如果执行模型是：

- 每个 token 位置都遍历一遍 operator queue。
- 每个 operator 后都 barrier。

那么 barrier 次数近似为：

$$
\text{barrier count} \approx \text{sequence length} \times \text{operator count}
$$

当 operator 很多、sequence 很长时，这会快速放大同步开销。

### 2.3 centralized spin barrier 容易形成共享热点

当前 barrier 有两个全局热点：

- `arrived`
- `generation`

每个 phase 所有线程都围绕这两个 cache line 竞争和轮询，容易产生：

- cache coherence traffic
- 慢线程拖尾
- 高核数下的原子争用

这不是实现错误，而是这种 barrier 结构的天然代价。

### 2.4 小 operator 会让同步成本失衡

对于轻量算子，例如某些 map / zip / elementwise operator，如果：

- 计算量较小
- 数据块较小
- 每次执行时间短

则可能出现“计算成本低于 barrier 成本”的情况。这样 runtime 更像是在频繁同步，而不是高效计算。

## 3. 设计目标

新的 runtime 同步设计需要满足以下目标：

1. 保留 persistent workers 架构，不强制改成每个 operator 临时建线程。
2. 降低 operator 间显式 barrier 的次数。
3. 把“当前轮工作完成”的语义从对称 barrier 改成非对称 join/latch，更贴近 `parallel_for` 的返回语义。
4. 为后续 operator fusion、任务批处理、调度器主导控制流留下空间。

## 4. barrier 和 join 的区别

### 4.1 Barrier

Barrier 的语义是：

- 参与者必须全部到达同步点。
- 所有参与者一起继续执行下一阶段。

这是对称同步。每个工作线程既是执行者，也是等待者。

适合场景：

- 多线程共同推进同一个 phase。
- phase 之间要求严格对齐。
- 所有线程下一步都必须立刻进入同一阶段。

### 4.2 Join / Latch

Join 的语义是：

- 一方发布任务。
- 多个 worker 并行执行。
- 发布方等待“这轮任务全部完成”。
- 是否进入下一轮，由发布方决定。

这是非对称同步。worker 不需要彼此 barrier，只需要上报完成。

适合场景：

- 存在 scheduler / coordinator 主导整体流程。
- worker 只是执行单元。
- 当前轮结束后是否进入下一轮，由调度器统一控制。

## 5. 推荐方案：引入 Epoch Join / Countdown Latch

### 5.1 总体思路

保留 persistent workers，但把“每个 operator 后全体 barrier”改为“每轮 operator 由调度线程发起，worker 完成后调度线程 join”。

核心组件：

- `epoch`：标识当前是第几轮 operator 调度。
- `remaining`：当前轮还剩多少 worker 未完成。
- `job descriptor`：描述当前轮要执行的 operator 或 operator group。
- `coordinator`：负责发布任务并等待 join 完成。

### 5.2 执行流程

每一轮执行流程如下：

1. coordinator 准备当前轮的 job。
2. coordinator 推进 `epoch`。
3. coordinator 把 `remaining` 设为参与 worker 数。
4. worker 读取当前 job，并按 `thread_id` 处理自己的数据分片。
5. worker 完成本轮后，对 `remaining` 做递减。
6. 最后一个完成的 worker 负责通知 coordinator。
7. coordinator 在确认本轮完成后，再发布下一轮 job。

这个流程中，worker 之间不必在每轮末尾互相 barrier。

### 5.3 为什么要加 epoch

如果 runtime 是长期运行的，单纯依赖一个计数器会有“上一轮完成信号和下一轮串台”的风险。

引入 `epoch` 的意义：

- 明确地区分第 $n$ 轮和第 $n+1$ 轮。
- worker 能知道自己当前处理的是哪一轮任务。
- coordinator 能准确等待某个特定 epoch 完成。

### 5.4 为什么这个模型更接近 `parallel_for`

`parallel_for` 的调用者语义其实是：

- 发起一轮并行工作。
- 等待这一轮工作全部结束。
- 返回。

这更像 join/latch，而不是 worker 之间的显式 barrier。

因此，对 eLLM 来说，epoch join 比统一 barrier 更接近高性能 CPU runtime 的控制流。

## 6. 方案落地建议

### 6.1 先保留 barrier，但减少使用范围

短期内不需要立刻完全删除 barrier，可以先做以下收缩：

1. 不再要求每个轻量 operator 后都 barrier。
2. 对可以连续执行的 operator 做 group 化，把多个 operator 合成一个 execution phase。
3. 只在真正有跨线程阶段依赖的地方保留 barrier。

### 6.2 区分三类 operator

#### A. 必须保留 phase barrier 的 operator

满足以下条件的 operator 可以继续使用 barrier：

- 当前 operator 产出的共享状态会被所有线程下一阶段立即读取。
- 下一阶段开始前，必须确保所有线程都完成当前阶段。
- 存在明显的 phase 依赖，而不是单纯的数据分片独立执行。

例如：

- 写共享 scratch buffer，然后下一阶段全线程读取。
- 需要全局聚合结果后再进入下一阶段。

#### B. 更适合 join 的 operator

满足以下条件的 operator，更适合使用 join/latch：

- 每个线程只处理自己负责的数据块。
- operator 完成后，只有 coordinator 需要知道“这一轮结束了”。
- worker 不需要立刻同步进入同一步。

例如：

- 普通 matmul 分块执行。
- map/zip 类按块并行算子。
- 每线程输出写入自己独占区域的算子。

#### C. 最应该融合的 operator

如果某个 operator 很轻，而且总是和前后 operator 成对出现，那么优先考虑 fusion，而不是保留单独 barrier。

典型候选：

- elementwise map + zip
- add + norm
- gate + activation + mul
- 小的 reshape / copy / 简单后处理

目标是增大“每次同步之前的有效工作量”。

## 7. 不建议的方向

### 7.1 不建议只优化 barrier 原子实现，而不改粒度

如果 barrier 仍然是：

- 每个 operator 一次
- 每个 token 都重复

那么即使 barrier 实现再快，整体收益也有限。真正的问题是同步过于频繁。

### 7.2 不建议所有 operator 一律全线程参与

对较小任务、较轻 operator，强制所有线程参与往往会适得其反。高性能 runtime 一般会根据任务规模决定：

- 是否全线程参与
- 是否只用部分线程
- 是否直接单线程执行

### 7.3 不建议把 barrier 和 join 混成一个概念

Barrier 和 join 都能表达“等待完成”，但职责不同：

- barrier 是 phase 对齐工具。
- join 是任务完成通知工具。

runtime 设计中要清楚地区分它们，否则后续调度逻辑会越来越难维护。

## 8. 推荐的演进路线

### Phase 1：减少 barrier 频度

- 统计当前 operator queue 中每轮 barrier 次数。
- 标出轻量 operator。
- 合并可以连续执行的 operator。
- 先把 barrier 次数降下来。

### Phase 2：引入 coordinator + epoch join

- 由 coordinator 发布当前轮 job。
- worker 做自己那份任务。
- 用 countdown latch / join counter 表示“本轮完成”。
- 由 coordinator 决定何时进入下一轮。

### Phase 3：按 operator 类型选择同步模型

- 真正的 phase 依赖，继续保留 barrier。
- 独立数据分片执行，改用 join/latch。
- 轻量算子，优先做 fusion。

### Phase 4：引入动态并行策略

- 对小任务只启用部分线程。
- 对极小任务直接串行。
- 对大任务再启用全线程。

这一步做完后，runtime 的行为会更接近高性能 CPU `parallel_for` 模型，而不是“所有阶段全核 rigid 同步”。

## 9. 总结

当前 eLLM 的 barrier 方案在语义上没有问题，但它更像 BSP 风格的 runtime phase barrier，而不是 `parallel_for` 那种 operator-internal join。

关键结论如下：

1. `parallel_for` 也要等待所有线程完成。
2. 真正的差别不在“等不等”，而在“在哪里等、等多频繁、每次等之前做了多少工作”。
3. 对 eLLM 当前的 persistent workers 架构，更合理的演进方向不是单纯优化 barrier，而是引入 epoch join / countdown latch，让 coordinator 主导 operator 轮次完成。
4. barrier 应保留给真正的 phase 依赖；普通 operator 完成通知应逐步转向 join/latch；轻量 operator 应优先融合。

因此，推荐路线是：

- 先减少 barrier 次数。
- 再引入 epoch join。
- 最后按 operator 类型混合使用 barrier / join / fusion。
