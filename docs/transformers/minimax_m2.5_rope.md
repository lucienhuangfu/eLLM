# MiniMax M2.5 RoPE 原理

## 1. 这份文档讲什么

这份文档说明 `MiniMax-M2.5` 的 Rotary Embedding 设计，并对齐到当前仓库里的实现方式。

重点回答四个问题：

1. `MiniMax-M2.5` 的 RoPE 和普通 RoPE 有什么关系
2. `rope_init_fn(self.config, device)` 为什么会返回 `inv_freq` 和 `attention_scaling`
3. `rotary_dim = 64` 在语义上代表什么
4. 当前仓库里这套逻辑是怎么落地的

---

## 2. 一句话结论

`MiniMax-M2.5` 使用的是 **部分维度 RoPE**。

它的含义不是“整条 head 维度都旋转”，而是：

1. 每个 head 的前 `rotary_dim = 64` 维进入 RoPE 子空间
2. 这个子空间内部仍然按相邻两维一组做 `cos/sin` 旋转
3. 剩余 `64` 维保留为非旋转内容通道

所以它可以看成是：

> `MiniMax-M2.5 = rotary subspace + standard pairwise RoPE + non-rotary residual channels`

---

## 3. MiniMax-M2.5 的关键配置

从 `models/MiniMax-M2.5/config.json` 看，和 RoPE 直接相关的字段主要是：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `head_dim` | `128` | 每个 attention head 的总维度 |
| `rotary_dim` | `64` | 参与 RoPE 的维度 |
| `rope_theta` | `5000000` | RoPE 的基频底数 |
| `use_qk_norm` | `true` | Q/K 先做归一化语义 |
| `qk_norm_type` | `per_layer` | QK Norm 的组织方式 |

这里最关键的是 `head_dim` 和 `rotary_dim` 的区别：

- `head_dim` 是完整 head 宽度
- `rotary_dim` 是真正进入旋转的子空间宽度

---

## 4. RoPE 的核心思想

RoPE 的目的不是给 token 再加一个普通位置向量，而是把位置信息直接写进 `Q/K` 的几何关系里。

它的基本做法是：

1. 把相邻两个标量维度看成一个复数对
2. 对这个复数对乘上与位置相关的复数旋转因子
3. 让 `Q` 和 `K` 的点积天然携带相对位置信息

对一对维度 `(x_even, x_odd)`，旋转形式可以写成：

```text
x' = x_even * cos(theta) - x_odd * sin(theta)
y' = x_even * sin(theta) + x_odd * cos(theta)
```

这就是典型的 pairwise RoPE。

---

## 5. `rope_init_fn` 返回了什么

你看到的 Python 代码：

```python
inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
```

说明 `rope_init_fn` 不只是生成频率表，它还会返回一个和该 RoPE 变体配套的缩放系数。

### 5.1 `inv_freq`

`inv_freq` 负责描述每个 rotary pair 的频率。

对第 `i` 个 pair：

```text
angle(pos, i) = pos * inv_freq[i]
```

再由这个角度生成：

```text
cos(angle), sin(angle)
```

### 5.2 `attention_scaling`

`attention_scaling` 是 **RoPE 变体的缩放因子**，不是普通 attention 里的 `1 / sqrt(head_dim)`。

它的作用是把某些 RoPE 变体在应用时的数值尺度调回更稳定的范围。

可以把它理解成：

1. `inv_freq` 决定“怎么转”
2. `attention_scaling` 决定“转的时候要不要额外缩放”

在 Hugging Face 的 RoPE 体系里，这类东西通常会以 `attention_factor` 的形式出现，尤其常见于带缩放策略的 RoPE 变体。

---

## 6. MiniMax-M2.5 的 RoPE 是怎么工作的

### 6.1 先确定旋转子空间

`rotary_dim = 64` 表示每个 head 的前 `64` 维进入 RoPE。

这 `64` 维内部仍然会继续按两维一组做旋转，也就是：

1. 第 0、1 维组成一对
2. 第 2、3 维组成一对
3. 以此类推

### 6.2 再生成频率表

频率表的生成遵循普通 RoPE 的逻辑：

1. 根据 `rotary_dim` 和 `rope_theta` 算每个 pair 的频率
2. 对每个 position 算对应的角度
3. 预计算出整张 `cos/sin` cache

### 6.3 最后把旋转应用到 Q/K

实际执行时，`Q` 和 `K` 会取出对应 position 的 `cos/sin`，再做复数乘法。

也就是说：

1. cache 先准备好
2. 运行时只做查表
3. 再把旋转结果写回

---

## 7. `attention_scaling` 在这里该怎么理解

如果只看 `MiniMax-M2.5` 的语义，`attention_scaling` 可以理解成 RoPE 的“配套缩放策略”。

更具体一点：

1. 它不是标准 attention logits 的缩放
2. 它不是 `QK` 点积后的 softmax 温度
3. 它是 RoPE 初始化阶段返回的、和频率表配套的参数

所以这句话最准确：

> `rope_init_fn` 返回的是“频率 + 缩放策略”，不是单纯的频率表。

如果你要把它和实现对齐，应该优先去看原始 Python 代码里 `attention_scaling` 最终乘到了哪里：

- 乘在 `inv_freq` 上
- 乘在 `cos/sin` cache 上
- 还是在后续 RoPE 应用阶段乘到 `Q/K` 上

这三种写法在语义上相近，但落点不同。

---

## 8. QK Norm 和 RoPE 的关系

`MiniMax-M2.5` 还开启了：

```json
"use_qk_norm": true
```

这说明 `Q` 和 `K` 在进入 attention 之前还会先做归一化语义。

可以把整个顺序理解为：

1. `Q/K` 先做归一化，保证数值稳定
2. 再应用 RoPE，把位置信息注入相位关系
3. 最后进入 attention score 计算

所以：

- `QK Norm` 解决的是尺度稳定性
- `RoPE` 解决的是位置信息注入

它们是串联关系，不是替代关系。

---

## 9. 当前仓库里的对应实现

### 9.1 RoPE cache 的生成

当前仓库里，RoPE 预计算逻辑在：

- [src/transformer/rope.rs](D:\eLLM\src\transformer\rope.rs)

核心函数是：

- `inv_freqs(dim, theta)`
- `precompute_freqs_cis(dim, max_sequence_length, theta)`
- `precompute_freqs_cis_t<T>(dim, max_sequence_length, theta)`

它做的事情很直接：

1. 先生成逆频率
2. 再按 position 生成 `cos/sin`
3. 把结果按线性数组缓存起来

### 9.2 模型初始化时挂载缓存

在 [src/transformer/model.rs](D:\eLLM\src\transformer\model.rs) 里，模型会把 RoPE cache 作为 `position_embedding` 建起来。

对应调用是：

```rust
precompute_freqs_cis_t::<f32>(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta as f32,
)
```

### 9.3 旋转是怎么执行的

真正的旋转动作主要落在：

- [src/kernel/scalar/complex_mul.rs](D:\eLLM\src\kernel\scalar\complex_mul.rs)
- [src/kernel/x86_64/f16_512/matmul_rms_complex.rs](D:\eLLM\src\kernel\x86_64\f16_512\matmul_rms_complex.rs)
- [src/operators/matmul/matmul3.rs](D:\eLLM\src\operators\matmul\matmul3.rs)

其中 `complex_mul` 做的就是标准复数乘法：

1. 读取一对实部/虚部
2. 读取对应的 `cos/sin`
3. 写回旋转后的结果

---

## 10. 这套实现的边界

当前仓库里的 `rope.rs` 是通用 cache 生成器，它本身并不关心某个模型是不是只用 `rotary_dim` 的一部分通道。

所以在接入 `MiniMax-M2.5` 时，要明确两件事：

1. 哪些通道进入 RoPE 子空间
2. `attention_scaling` 最终在哪一步生效

换句话说：

- `rotary_dim` 决定“旋转哪些通道”
- `attention_scaling` 决定“这套 RoPE 的缩放策略”

这两个职责不要混成一个概念。

---

## 11. 最小心智模型

如果把 `MiniMax-M2.5` 的 RotaryEmbedding 压缩成一句话，可以记成：

> 对每个 head 的 `rotary_dim` 子空间，先生成位置频率表，再按 position 做 pairwise 复数旋转，并在需要时附带 RoPE 变体的缩放因子。

更短一点就是：

1. 先算 `inv_freq`
2. 再算 `cos/sin`
3. 再用这些值旋转 `Q/K`
4. 如果 RoPE 变体需要，就再应用 `attention_scaling`

---

## 12. 结论

`MiniMax-M2.5` 的 RotaryEmbedding 可以理解为：

1. 标准 pairwise RoPE 的一种变体接入方式
2. 只在 `rotary_dim = 64` 的子空间内做旋转
3. 通过 `rope_init_fn` 同时拿到频率表和缩放策略

在当前仓库里，这套逻辑主要由 `rope.rs`、`complex_mul` 和 attention 路径共同完成。
