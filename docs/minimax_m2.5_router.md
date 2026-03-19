# MiniMax M2.5 Router 方案

## 1. 目标

这份文档只讨论 `MiniMax-M2.5` 的 router 版本设计，不包含代码实现。

目标是把当前 `SparseMoeRouter` 从“通用 MoE 路由器”收敛成一条明确的 `MiniMax M2.5 Router` 方案，便于后续接入：

1. `num_local_experts = 256`
2. `num_experts_per_tok = 8`
3. `scoring_func = sigmoid`
4. `use_routing_bias = true`
5. `gate` 不参与 fp8 转换

从当前仓库里的 `models/MiniMax-M2.5/config.json` 看，这个模型族的 router 不是简单的 softmax top-k 变体，而是带有 sigmoid 打分和 routing bias 的门控版本。

---

## 2. 版本定义

建议把 `MiniMax-M2.5` 的 router 版本定义为：

**Router V1: Gate Linear + Sigmoid Scoring + TopK Dispatch**

含义如下：

1. `hidden_states` 先经过 gate 投影
2. gate 输出不直接按普通 softmax 解释
3. router 分数使用 sigmoid 风格的打分语义
4. 取 top-k expert 做稀疏分发
5. 如配置启用，则加入 routing bias

这个版本适合当前仓库里“先做可运行骨架，再按模型族细化”的路线。

---

## 3. 与当前 `router.rs` 的关系

当前 `src/transformer/sparse_moe/router.rs` 的行为可以概括为：

1. `hidden_states.matmul(gate_weight)`
2. 调用 `experts_softmax_norm`
3. 输出路由相关的四元组指针

它的语义更偏向“通用 softmax router”。

而 `MiniMax M2.5` 需要的 router 版本应该额外具备：

1. sigmoid scoring 语义
2. routing bias
3. 明确的 expert budget 约束
4. 对 `num_local_experts = 256` 的本地专家组织方式

所以这不是简单复用现有 router 的参数，而是应当把它视为一个独立 router 版本。

---

## 4. 已知配置约束

基于本地配置文件，可稳定确认的约束有：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `hidden_size` | 3072 | router 输入维度 |
| `num_attention_heads` | 48 | 不是 router 本体参数，但说明模型规模 |
| `num_key_value_heads` | 8 | 不是 router 本体参数，但说明模型结构 |
| `num_hidden_layers` | 62 | router 会在所有层中重复出现 |
| `num_local_experts` | 256 | 本地 expert 总数 |
| `num_experts_per_tok` | 8 | 每个 token 选择的 expert 数 |
| `scoring_func` | `sigmoid` | router 的打分语义 |
| `use_routing_bias` | `true` | 需要 bias 参与路由 |
| `tie_word_embeddings` | `false` | 与 router 无直接关系 |
| `qk_norm_type` | `per_layer` | 与 attention 相关，不是 router 核心参数 |

其中最关键的是前三个：

1. `num_local_experts = 256`
2. `num_experts_per_tok = 8`
3. `scoring_func = sigmoid`

这三项决定了 router 的版本边界。

---

## 5. Router 输入输出契约

### 输入

router 只接受两类输入语义：

1. `hidden_states`
2. router 相关的配置和命名信息

其中 `hidden_states` 是 token 级别的表示，router 通过它计算每个 token 对各 expert 的偏好。

### 输出

router 输出应服务于后续 MoE dispatch，语义上需要包含：

1. token 到 expert 的选择结果
2. token 到 expert 的路由权重
3. token 分发/回收所需的索引信息
4. 与 top-k 选择一致的归一化信息

如果沿用当前仓库的指针式返回方式，那么输出仍然可以是路由相关的低层数组；但从方案层面看，核心是这四类信息必须完整。

---

## 6. Router 计算流程

### 6.1 Gate 投影

第一步是把 `hidden_states` 投影到 expert 维度。

对 `MiniMax M2.5` 来说，这一步的含义不是“普通分类头”，而是“token 对全部 local experts 的打分入口”。

### 6.2 Sigmoid scoring

gate 输出不能只按普通 softmax 去理解。

这里建议把 sigmoid 视为 router 的主语义：

1. gate 输出先变成每个 expert 的独立激活倾向
2. 再基于 expert budget 做 top-k 选择
3. 选择结果再进入 dispatch

这样可以保留 `sigmoid` 在多专家门控里的“独立打分”风格。

### 6.3 TopK 选择

模型给出的 `num_experts_per_tok = 8` 表明，每个 token 需要选择 8 个 expert。

因此 router 的版本必须明确：

1. 先排序或筛选 expert 分数
2. 取前 8 个
3. 输出对应权重和索引

这一步不是可选项，而是 `MiniMax M2.5` router 的核心契约。

### 6.4 Routing bias

`use_routing_bias = true` 表示 routing bias 是 router 版本的一部分，不是外围修饰。

文档层面的建议是：

1. bias 应在 expert 打分阶段介入
2. bias 应影响 top-k 的最终选择
3. bias 不能只做输出后处理

否则会破坏 router 的选择语义。

---

## 7. Expert 组织方式

`num_local_experts = 256` 表示 router 面对的是本地专家池，而不是概念上的单个专家列表。

因此 router 版本需要显式支持：

1. expert index 在本地池内闭包
2. token 的 top-k 结果落在 `[0, 255]`
3. dispatch 逻辑按本地 expert 池做分发

这意味着 router 的编号体系应尽量稳定，不要把本地 expert 编号和全局模型编号混在一起。

---

## 8. 与 fp8/量化的关系

本地配置里有一条很重要的约束：

`modules_to_not_convert = ["gate", "e_score_correction_bias", "lm_head"]`

这意味着 router 相关模块在量化路径里应被保护，不应被当作普通可转换模块处理。

方案上建议把它解释为：

1. gate 保持高精度
2. routing bias 保持高精度
3. router 的数值稳定性优先于压缩率

这也是 `MiniMax M2.5` router 版本需要单独定义的原因之一。

---

## 9. 推荐的文档级版本命名

为了后续在代码和文档里保持一致，建议采用下面的版本命名：

1. `RouterV1`：`sigmoid + topk + routing bias`
2. `MiniMaxM2RouterV1`：MiniMax M2.5 专用版本
3. `SparseMoeRouterV1`：当前通用版本，保留给不带 sigmoid/bias 特性的模型

如果后面再遇到其他 MiniMax 变体，可以继续往后演进：

1. `MiniMaxM2RouterV2`
2. `MiniMaxM25RouterV1`

这样文档和实现都能保持版本边界清楚。

---

## 10. 接入建议

如果目标是让当前仓库尽快支持 `MiniMax M2.5`，建议按下面顺序接入：

1. 先把 router 版本分出来，不和现有通用 router 混用
2. 再把 `sigmoid scoring` 和 `routing bias` 作为独立配置项描述
3. 然后补上 `num_local_experts = 256` 和 `num_experts_per_tok = 8`
4. 最后再决定是否需要和现有 `SparseMoeRouter` 共享底层 dispatch 逻辑

这里的原则是：

1. 版本先清晰
2. 语义先明确
3. 复用后置

这样做的好处是不会把“通用 MoE router”和“MiniMax M2.5 router”混成一类。

---

## 11. 不建议的做法

1. 不建议直接把现有 `softmax router` 当成 `MiniMax M2.5` 的完整实现
2. 不建议把 routing bias 只做成后处理补丁
3. 不建议把 gate 量化路径和普通线性层完全等同
4. 不建议把 `num_local_experts` 和 `num_experts_per_tok` 藏到散落的常量里

这些做法会让后续调试很难定位 router 行为。

---

## 12. 结论

`MiniMax M2.5` 的 router 更适合被定义为一个独立版本，而不是当前 `SparseMoeRouter` 的简单参数化。

推荐的文档结论是：

> `MiniMaxM2RouterV1 = gate linear + sigmoid scoring + routing bias + top-k dispatch`

它对应的关键配置是：

1. `num_local_experts = 256`
2. `num_experts_per_tok = 8`
3. `scoring_func = sigmoid`
4. `use_routing_bias = true`

如果你要，我下一步可以继续给你补一份同风格的 `docs/minimax_m2.5_moe_block.md`，把 router 和 expert dispatch 的配套方案也写完整。
