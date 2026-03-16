# 基于 `decode_list` 的全局索引反查方案

## 目标

已知一个本轮的全局 token 下标 `global_index`，需要反向查到：

* 这个 token 属于哪条 sequence
* 它在该 sequence 内的序列位置 `sequence_index`

当前可直接利用的数据结构就是：

```rust
pub struct SequenceSlice {
	pub token_start_index: usize,
	pub batch_index: usize,
	pub sequence_index: usize,
	pub length: usize,
}
```

其中 `decode_list: Vec<SequenceSlice>` 满足两个关键性质：

1. `decode_list` 按 sequence 的调度顺序依次 push。
2. 每个 slice 对应一个连续的全局区间：

$$
[token\_start\_index,\ token\_start\_index + length)
$$

另外，`length` 的语义和当前调度阶段直接相关：

* prefill 时，`length` 是这条 sequence 本轮实际进入全局 token 视图的连续长度，可能大于 `1`
* decode 时，每条 sequence 本轮只生成一个 token，因此 `length` 固定为 `1`

因此，反查问题本质上就是：

```text
给定 global_index，找到覆盖它的 SequenceSlice。
```

---

## 核心结论

如果命中的 slice 为：

```text
slice = {
	token_start_index,
	batch_index,
	sequence_index,
	length,
}
```

则局部偏移量是：

$$
offset = global\_index - slice.token\_start\_index
$$

对应的 sequence 内位置就是：

$$
local\_sequence\_index = slice.sequence\_index + offset
$$

最终返回：

* `batch_index`: 属于哪条 sequence
* `local_sequence_index`: 该 token 在该 sequence 中的真实位置
* `offset`: 它在当前 slice 内的偏移

---

## 推荐方案

推荐使用混合方案：

* 每个线程处理自己的 `global_index` 连续区间时，首个 index 用二分定位
* 后续 index 用当前 `slice_index` 顺序推进

而不是对区间里的每个 `global_index` 都单独做一次二分。

原因：

* `decode_list` 本身已经是按全局 token 顺序排好的区间表。
* `token_start_index` 单调递增。
* 单个 slice 覆盖的是连续区间，适合二分定位。
* 多线程场景下，每个线程拿到的通常是一个连续的全局区间，首点定位后继续顺推更便宜。
* 不需要额外维护 `global_index -> sequence` 的大数组。

### 接口建议

```rust
pub struct DecodeLookupResult {
	pub slice_index: usize,
	pub batch_index: usize,
	pub sequence_index: usize,
	pub offset_in_slice: usize,
}

pub fn lookup_decode_index(
	decode_list: &[SequenceSlice],
	global_index: usize,
) -> Option<DecodeLookupResult>
```

其中：

* `slice_index` 表示命中了 `decode_list` 的第几个 slice
* `batch_index` 用来标识是哪条 sequence
* `sequence_index` 是反查后的 sequence 内位置
* `offset_in_slice` 方便后续算子继续用

---

## 查找逻辑

### 1. 边界检查

先用最后一个 slice 判断 `global_index` 是否越界：

$$
max\_token = last.token\_start\_index + last.length
$$

若：

$$
global\_index \ge max\_token
$$

则直接返回 `None`。

### 2. 首个 index 二分定位 slice

在 `decode_list` 上找满足下面条件的 slice：

$$
slice.token\_start\_index \le global\_index < slice.token\_start\_index + slice.length
$$

单点查询时，可以直接写成标准二分：

```rust
pub fn lookup_decode_index(
	decode_list: &[SequenceSlice],
	global_index: usize,
) -> Option<DecodeLookupResult> {
	let last = decode_list.last()?;
	if global_index >= last.token_start_index + last.length {
		return None;
	}

	let mut left = 0usize;
	let mut right = decode_list.len();

	while left < right {
		let mid = (left + right) / 2;
		let slice = &decode_list[mid];

		if global_index < slice.token_start_index {
			right = mid;
			continue;
		}

		let slice_end = slice.token_start_index + slice.length;
		if global_index >= slice_end {
			left = mid + 1;
			continue;
		}

		let offset_in_slice = global_index - slice.token_start_index;
		return Some(DecodeLookupResult {
			slice_index: mid,
			batch_index: slice.batch_index,
			sequence_index: slice.sequence_index + offset_in_slice,
			offset_in_slice,
		});
	}

	None
}
```

这适合：

* 随机查询一个 `global_index`
* 每个线程处理区间时，先定位自己的起始 `global_index`

### 3. 后续 index 用游标顺推

如果一个线程拿到的是连续区间：

$$
[global\_begin,\ global\_end)
$$

则没必要对这个区间里的每个下标反复二分。

更合适的流程是：

1. 对 `global_begin` 做一次二分，找到起始 `slice_index`
2. 维护当前 slice 的结束位置：

$$
slice\_end = slice.token\_start\_index + slice.length
$$

3. 当 `global_index < slice_end` 时，继续使用当前 slice
4. 当 `global_index >= slice_end` 时，`slice_index += 1`，切到下一个 slice

伪代码如下：

```rust
pub fn walk_decode_range(
	decode_list: &[SequenceSlice],
	global_begin: usize,
	global_end: usize,
	mut visit: impl FnMut(usize, usize, usize),
) {
	let Some(mut found) = lookup_decode_index(decode_list, global_begin) else {
		return;
	};

	let mut slice_index = found.slice_index;

	for global_index in global_begin..global_end {
		while slice_index < decode_list.len() {
			let slice = &decode_list[slice_index];
			let slice_end = slice.token_start_index + slice.length;
			if global_index < slice_end {
				let offset = global_index - slice.token_start_index;
				visit(global_index, slice.batch_index, slice.sequence_index + offset);
				break;
			}
			slice_index += 1;
		}
	}
}
```

这里 `visit(global_index, batch_index, sequence_index)` 表示：

* 当前访问到的全局位置
* 它对应的 batch 槽位
* 它对应的 sequence 内位置

---

## 为什么这个方案成立

### Decode 场景

在 decode 轮里，当前调度器写入的是：

```text
length = 1
token_start_index = decode_list 中的顺序下标
sequence_index = kv_index
```

所以此时反查会退化成：

```text
global_index == decode_list[global_index] 对应的那条 sequence
```

也就是说 decode 轮几乎不需要真正二分，直接按下标访问都可以。

### Prefill 场景

在 prefill 轮里，`decode_list` 中每个 slice 代表该 sequence 本轮的一段连续 token 区间，此时：

```text
length = 该 sequence 本轮被调度进去的真实 token 数
```

也就是说，prefill 时 `length` 不是 `1`，而是具体长度；只有 decode 时 `length` 才固定为 `1`。

例如：

```text
decode_list[0] = { batch=0, sequence_index=0, token_start_index=0, length=6 }
decode_list[1] = { batch=1, sequence_index=0, token_start_index=6, length=2 }
```

那么：

* `global_index = 0..5` 属于 batch 0
* `global_index = 6..7` 属于 batch 1

若 `global_index = 7`，则：

$$
offset = 7 - 6 = 1
$$

$$
sequence\_index = 0 + 1 = 1
$$

因此可反推出它对应的是：

```text
batch_index = 1
sequence_index = 1
```

---

## 如果查的是“第几个 sequence”

如果业务里说的不是 `batch_index`，而是“它是本轮第几个排进 `decode_list` 的 sequence”，那么直接返回：

```text
sequence_order = slice_index
```

前提是当前 `decode_list` 仍保持“一条 sequence 在该列表里只出现一次”。

按当前调度器实现，这个前提成立：

* decode 轮，每条 sequence 只 push 一个 `length=1` 的 slice
* prefill 轮，每个 candidate 只 push 一个汇总 slice 到 `decode_list`

因此：

* `slice_index` = 本轮 sequence 顺序编号
* `batch_index` = batch 槽位编号
* `sequence_index` = sequence 内 token 位置

这三者需要区分清楚，不要混用。

---

## 复杂度

直接二分方案：

* 时间复杂度：$O(\log N)$
* 空间复杂度：$O(1)$

其中 $N = decode\_list.len()$。

---

## 复杂度分析

### 单点随机查询

直接二分：

* 时间复杂度：$O(\log N)$
* 空间复杂度：$O(1)$

其中 $N = decode\_list.len()$。

### 多线程连续区间查询

假设一个线程负责：

$$
K = global\_end - global\_begin
$$

则更合理的成本是：

* 首次定位：$O(\log N)$
* 区间内顺推：$O(K + S)$

其中 $S$ 是这个线程处理区间内实际跨过的 slice 数。

因为 `slice_index` 只会单调递增，不会回退，所以这是比“每个下标都二分一次”更符合线程分段场景的做法。

当每个线程都拿到连续区间时，总体策略可以概括为：

```text
每线程第一次查找用二分；
之后沿 decode_list 顺序推进。
```

---

## 最终建议

当前版本先不要新增额外索引结构，直接采用下面这条规则：

```text
用 decode_list 作为全局 token 区间表；
每个线程对自己负责区间的第一个 global_index 做一次二分；
后续 index 沿当前 slice 顺推；
必要时切到下一个 slice；
再用 sequence_index + (global_index - token_start_index)
反推出该 token 的 sequence 内位置。
```

这是和现有 `BatchScheduler` 以及多线程分段执行方式最一致、改动最小、可直接落地的方案。
