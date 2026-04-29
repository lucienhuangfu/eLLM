# MiniMax M2.5 Router 方案

## 1. 文档目的

这份文档说明 `MiniMax-M2.5` 的 router 设计，并对齐当前实现的结构。

重点不是贴代码，而是回答三个问题：

1. 这个 router 为什么不能按普通 softmax MoE 理解
2. 现在的实现把哪些 `operators` 合并在一起了
3. 这些合并后的算子如何串成完整的数据流

---

## 2. Router 目标

`MiniMax-M2.5` 的 router 需要满足这些约束：

1. `num_local_experts = 256`
2. `num_experts_per_tok = 8`
3. `scoring_func = sigmoid`
4. `use_routing_bias = true`
5. `gate` 不参与 fp8 转换

这说明它不是一个普通的 `softmax top-k` router，而是带有独立打分语义和 routing bias 的门控路由。

因此，文档里把它定义为：

> `MiniMaxM2RouterV1 = gate linear + sigmoid scoring + routing bias + top-k dispatch`

---

## 3. 总体结构

当前实现不是一个 operator 从头做到底，而是拆成两段：

1. `ExpertsSigmoidGate` 负责生成 expert 分数
2. `ExpertsTopkNorm` 负责把分数转换成最终路由结果

也就是说，router 的处理链路可以概括为：

`hidden_states -> gate scoring -> top-k norm -> dispatch buffers`

这就是当前结构最重要的边界。

---

## 4. operator 合并方式

这次代码改动的核心，是把原来分散的 gate 逻辑收拢到一个独立 operator 里。

### 4.1 合并了什么

`ExpertsSigmoidGate` 现在承担的不是单一的矩阵乘法，而是三件事的组合：

1. 线性投影
2. routing bias 注入
3. sigmoid 激活

这意味着 `gate` 这一步已经从“调用 matmul 后再补逻辑”，变成了一个完整的门控算子。

### 4.2 没有合并什么

有些能力仍然保持分离：

1. `MatMul` 仍然是底层块乘能力的提供者
2. `ExpertsTopkNorm` 仍然负责 top-k 选择和归一化
3. dispatch buffer 的组织仍然由 router 上层维护

所以这里的“合并”不是把所有逻辑揉成一个大函数，而是把最紧密的一段逻辑做成了一个语义完整的 operator。

### 4.3 合并的价值

这样拆分以后有三个直接好处：

1. `gate` 的语义更清楚，不再依赖通用 `MatMul` 容器
2. `routing bias` 不再是后处理补丁，而是计算链路的一部分
3. kernel 只需要面向 gate 的语义，不需要知道上层 router 的完整形态

---

## 5. 计算流程

### 5.1 Gate 投影

`hidden_states` 先进入 gate 投影，得到每个 token 对全部 local experts 的原始分数。

这里的分数不是普通分类头的输出，而是 router 的入口分数。

### 5.2 Sigmoid scoring

当前设计把 `sigmoid` 当成 router 的主语义。

它表达的是每个 expert 的独立激活倾向，而不是 softmax 那种全局竞争式概率。

### 5.3 Routing bias

`use_routing_bias = true` 表示 bias 必须参与打分阶段，而不是在 top-k 之后补上。

这样 bias 才会真正影响 expert 的排序和选择。

### 5.4 Top-k 选择

`num_experts_per_tok = 8` 决定了每个 token 需要保留 8 个 expert。

因此后续的 `ExpertsTopkNorm` 负责：

1. 选出 top-8 expert
2. 生成对应权重
3. 生成索引和指示信息

---

## 6. 代码层职责

### 6.1 `ExpertsSigmoidGate`

这个 operator 负责 gate 计算本身，内部已经包含：

1. 输入和权重指针
2. 输出缓冲区
3. tile 调度参数
4. 线程私有的 panel 池

它的职责是把 gate 这一步算完整，而不是只调用一次普通 matmul。

### 6.2 `MatMulTrait`

`MatMulTrait` 保留为底层乘法接口。

它的角色不是承载 router 语义，而是给 gate 提供复用的块乘能力和统一的计算入口。

### 6.3 `ExpertsSigmoidGateTrait`

这个 trait 把 gate 的语义挂到乘法路径上。

它的作用是说明：这里不是普通矩阵乘法，而是带 sigmoid router 语义的 gate 计算。

### 6.4 `ExpertsTopkNorm`

这个 operator 负责 gate 输出之后的整理工作：

1. top-k 选择
2. 权重归一化
3. expert 指示位和索引输出

它和 `ExpertsSigmoidGate` 是串联关系，不是替代关系。

---

## 7. 数据流契约

### 输入

router 的输入主要有两类：

1. `hidden_states`
2. gate 权重和路由配置

其中 `hidden_states` 是 token 级别表示，router 会据此计算每个 token 对 expert 的偏好。

### 输出

router 最终要提供给 MoE dispatch 的信息包括：

1. token 到 expert 的选择结果
2. token 到 expert 的路由权重
3. token 分发和回收需要的索引
4. top-k 对应的归一化信息

当前实现中，这些信息是由 `ExpertsSigmoidGate` 和 `ExpertsTopkNorm` 分两步完成的。

---

## 8. 配置约束

下面这些配置是当前 router 版本的关键边界：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `hidden_size` | 3072 | router 输入维度 |
| `num_hidden_layers` | 62 | router 会在所有层中重复出现 |
| `num_local_experts` | 256 | 本地 expert 总数 |
| `num_experts_per_tok` | 8 | 每个 token 选择的 expert 数 |
| `scoring_func` | `sigmoid` | router 的打分语义 |
| `use_routing_bias` | `true` | bias 参与路由 |
| `tie_word_embeddings` | `false` | 与 router 无直接关系 |
| `qk_norm_type` | `per_layer` | attention 相关，不是 router 核心参数 |

这里最关键的是：

1. `num_local_experts = 256`
2. `num_experts_per_tok = 8`
3. `scoring_func = sigmoid`

这三项决定了 router 的版本边界。

---

## 9. 量化约束

配置里还有一条重要信息：

`modules_to_not_convert = ["gate", "e_score_correction_bias", "lm_head"]`

这说明 router 相关模块不应该走普通量化转换路径。

文档上可以把它理解为：

1. gate 保持高精度
2. routing bias 保持高精度
3. router 的数值稳定性优先于压缩率

---

## 10. 推荐命名

为了保持代码和文档一致，建议使用下面的命名方式：

1. `RouterV1`：`sigmoid + topk + routing bias`
2. `MiniMaxM2RouterV1`：MiniMax M2.5 专用版本
3. `SparseMoeRouterV1`：通用 softmax router 版本

如果后面继续演进，可以再增加：

1. `MiniMaxM2RouterV2`
2. `MiniMaxM25RouterV1`

---

## 11. 结论

`MiniMax-M2.5` 的 router 更适合被视为一个独立版本，而不是 `SparseMoeRouter` 的简单参数化。

当前实现的关键点可以总结为：

1. `gate` 已经合并了 `matmul + routing bias + sigmoid`
2. `router` 上层仍然保留 `gate -> topk` 的两段式结构
3. `ExpertsTopkNorm` 负责把分数翻译成 dispatch 需要的最终结果

如果只保留一句话：

> `MiniMaxM2RouterV1 = gate linear + sigmoid scoring + routing bias + top-k dispatch`
