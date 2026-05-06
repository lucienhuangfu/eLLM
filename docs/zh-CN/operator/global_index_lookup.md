# 基于 `decode_list` 的全局索引反查方案

## 目标

已知一个本轮的全局 token 下标 `global_index`，需要反向查到：

* 这个 token 属于哪条 sequence
* 它在该 sequence 内的序列位置 `sequence_index`

当前可利用的数据结构是 `DecodeList`。它可以理解成一张“本轮全局 token 视图的区间表”。

这张表里的每一项 `SequenceSlice` 都描述了一段连续区间：

* 这段区间从哪个全局位置开始
* 这段区间属于哪个 batch 槽位
* 这段区间对应 sequence 的哪个起始位置
* 这段区间有多长

其中 `last_token_flag` 只是后续输出逻辑要用的标记，不参与全局索引反查。

---

## 核心思想

问题本质上只有一句话：

```text
给定 global_index，找到覆盖它的那一段 slice。
```

一旦找到了命中的 slice，剩下就只是算偏移。

如果某个 slice 表示：

* 这段区间从全局位置 `token_start_index` 开始
* 它对应 sequence 内位置 `sequence_index` 开始

那么对于这个区间中的任意一个全局位置：

$$
offset = global\_index - token\_start\_index
$$

于是它在 sequence 内的位置就是：

$$
sequence\_index + offset
$$

也就是说，反查分两步：

1. 先找到 `global_index` 落在哪个 slice 里
2. 再用相对偏移推回真实的 sequence 位置

---

## 可以把它想成什么

可以把 `decode_list` 想成把多条 sequence 在“本轮计算视图”里拼接起来之后形成的一张目录表。

简单示意：

```text
全局位置:     0 1 2 3 4 5 6 7
归属 slice:   A A A A A A B B
```

这等价于两段区间：

```text
slice A: [0, 6)
slice B: [6, 8)
```

如果：

* slice A 对应 batch 0，sequence 起点是 0
* slice B 对应 batch 1，sequence 起点是 0

那么：

* 全局位置 `4` 落在 A 中，对应 batch 0、sequence 位置 4
* 全局位置 `7` 落在 B 中，对应 batch 1、sequence 位置 1

这里的关键不是“它在第几个 token”，而是“它落在哪一段区间里”。

---

## 为什么 `decode_list` 能做这件事

`decode_list` 成立的前提，是它本身已经按全局 token 顺序组织好了。

也就是说：

* slice 的顺序就是全局 token 的顺序
* 每个 slice 覆盖一段连续区间
* 后一个 slice 的位置总在前一个 slice 后面

所以它天然适合做“区间反查”。

不需要额外建一张很大的数组，把每个 `global_index` 都显式映射回 sequence。

---

## 单点查询的原理

单点查询适合这种场景：

* 随机给一个 `global_index`
* 只查一次，或者次数不多

这时最自然的思路是：

* 在所有 slice 里，找到第一个“右边界大于 `global_index`”的区间
* 再确认 `global_index` 没有落在这个区间左边

如果命中，就能算出：

* 它属于哪个 `batch_index`
* 它在 sequence 内的真实位置
* 它命中了第几个 slice

如果没命中，说明：

* `global_index` 越界了
* 或者区间之间存在空洞，而这个位置正好落在空洞里

因此单点查询的本质是：

```text
先定位区间，再计算区间内偏移。
```

---

## 连续区间查询的原理

多线程执行时，更常见的情况不是“随机查一个点”，而是“一个线程负责一段连续全局区间”。

例如某个线程负责：

$$
[global\_begin,\ global\_end)
$$

这时如果对区间里的每个位置都重新查一次，会有很多重复工作。

更合理的做法是：

1. 先对起点 `global_begin` 做一次定位
2. 得到当前命中的 `slice_index`
3. 在当前 slice 内顺着往前走
4. 走到这个 slice 的末尾后，再切到下一个 slice

简单示意：

```text
线程负责区间: [4, 8)

全局位置:     0 1 2 3 4 5 6 7
归属 slice:   A A A A A A B B
线程访问:             ^ ^ ^ ^
```

这个线程只需要：

* 先确定位置 `4` 在 A 中
* 然后顺着访问 `5`
* 再切到 B，访问 `6`、`7`

也就是说，连续区间查询的核心不是“反复定位”，而是：

```text
第一次定位，后面顺推。
```

---

## Decode 场景为什么更简单

在 decode 轮里，每条 sequence 本轮只贡献一个 token。

因此每个 slice 的长度固定为 `1`，于是整张表几乎退化成：

```text
第 0 个全局位置 -> 第 0 个 slice
第 1 个全局位置 -> 第 1 个 slice
第 2 个全局位置 -> 第 2 个 slice
...
```

换句话说，decode 场景下区间表虽然还存在，但每段区间只有一个点，所以查找非常直接。

---

## Prefill 场景为什么需要区间思维

在 prefill 轮里，一条 sequence 本轮可能一次进入多个连续 token。

所以一个 slice 的长度可能大于 `1`。

这时 `decode_list` 更像下面这样：

```text
slice A 覆盖 [0, 6)
slice B 覆盖 [6, 8)
slice C 覆盖 [8, 12)
```

于是全局位置和 sequence 位置不再是一一对应，而必须通过“落在哪个区间里”来判断。

这正是 `lookup_global_index` 和 `walk_global_range` 要解决的问题。

---

## `slice_index` 的意义

反查结果里除了 `batch_index` 和 `sequence_index`，还会保留 `slice_index`。

它的作用不是表示 token 在 sequence 内的位置，而是表示：

```text
当前命中了 decode_list 里的第几个 slice
```

这个信息对连续区间顺推很有用，因为下一次不必重新从头找，只需要从当前 slice 往后继续走。

如果业务上问的是“这是本轮排进来的第几条 sequence”，那么在当前实现前提下，`slice_index` 也可以直接作为顺序编号理解。

---

## 复杂度为什么合理

单点查询时：

* 需要在区间表里定位一次
* 成本是 $O(\log N)$

连续区间查询时：

* 只在起点做一次定位
* 后面主要是在区间内顺着走

因此成本可以理解为：

* 起点定位：$O(\log N)$
* 区间遍历：$O(K + S)$

其中：

* $N$ 是 `decode_list` 的长度
* $K$ 是线程实际要处理的全局位置数
* $S$ 是这段区间里跨过的 slice 数

这比“每个位置都重新查一次”更适合当前按连续区间分工的线程模型。

---

## 最终建议

当前最合适的理解方式是：

```text
把 DecodeList 当成一张全局 token 区间表。
单点查询时，先找命中的区间，再算区间内偏移。
连续区间查询时，起点先定位，之后沿 slice 顺推。
```

这样既符合当前 `BatchScheduler` 的数据组织方式，也符合多线程按连续区间处理 token 的执行方式。
