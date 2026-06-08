# 新增算子

本指南介绍如何在 eLLM 中实现、集成和测试一个新算子。

---

## 1. 了解算子模型

eLLM 中的算子是一个计算单元，由 `initialize_serving_resources()` 放入算子队列，
并由 `ServingRunner` 在每个调度轮次执行。

关键类型和 trait：

- `Operator<T>` 枚举 — `src/operators/operator.rs` 中的分发枚举
- `run(thread_id, thread_num, task: &ScheduleTask)` — 每轮每个线程调用一次
- `ScheduleTask` — 携带本轮的 `prefill_list`、`decode_list` 和计数

算子对请求应该是无状态的。所有可变请求状态存在于 `BatchSequence` 和 `Vec<SequenceState>` 中。

---

## 2. 创建算子文件

在 `src/operators/` 的合适子目录下新建文件。

简单 elementwise 算子的骨架：

```rust
// src/operators/elementwise/my_op.rs

use crate::runtime::scheduling::types::ScheduleTask;

pub struct MyOp {
    // 构造时捕获的配置
}

impl MyOp {
    pub fn new(/* ... */) -> Self {
        MyOp { /* ... */ }
    }

    pub fn run(&self, thread_id: usize, thread_num: usize, task: &ScheduleTask) {
        // 使用 task.decode_list 或 task.prefill_list 确定本轮处理哪些 token。
        // 使用 assign(total, thread_num, thread_id) 分割线程工作。
    }
}
```

在父 `mod.rs` 中暴露：

```rust
pub mod my_op;
pub use my_op::MyOp;
```

---

## 3. 添加到算子枚举

在 `src/operators/operator.rs` 中添加变体：

```rust
pub enum Operator<T> {
    // ... 现有变体 ...
    MyOp(MyOp),
}

impl<T> Operator<T> {
    pub fn run(&self, thread_id: usize, thread_num: usize, task: &ScheduleTask) {
        match self {
            // ... 现有分支 ...
            Operator::MyOp(op) => op.run(thread_id, thread_num, task),
        }
    }
}
```

---

## 4. 将算子加入队列

在 `src/serving/resources.rs`（或算子队列组装处）推入新算子：

```rust
operator_queue.push(Operator::MyOp(MyOp::new(/* ... */)));
```

队列由 `ServingRunner` 每轮按顺序执行。

---

## 5. 编写对齐测试

推荐的验证工作流：

1. 在 `alignment/<op_name>/` 下编写 Python 脚本，使用参考实现（NumPy、PyTorch 等）
   计算期望输出并保存为 `.npy` 文件。
2. 编写 Rust 对齐二进制，加载相同输入，运行算子，并与保存的 `.npy` 输出比对。

参考示例：`alignment/silu_mul/` 和 `alignment/rope/`。

在 `Cargo.toml` 中添加入口：

```toml
[[bin]]
name = "my_op_alignment"
path = "alignment/my_op/my_op_alignment.rs"
```

运行对比：

```bash
python alignment/my_op/generate_data.py
cargo run --bin my_op_alignment
```

---

## 6. 命名规范

| 对象 | 规范 |
|------|------|
| 算子结构体 | `PascalCase`（如 `SiluMulZip`） |
| 算子文件 | `snake_case.rs`（如 `silu_mul_zip.rs`） |
| 对齐目录 | `alignment/<op_name>/` |
| 测试输入文件 | `python_<op_name>_<role>.npy` |

---

## 7. 参考

- [FakeEcho](../operator/fake_echo.md) — 最简单的算子，很好的参考
- [LiftVector](../operator/left_vector.md) — 最小内存移动算子
- [Attention](../operator/attention.md) — 完整静态并行算子
- [HuggingFace 对齐](../reference/hf_alignment.md) — golden-file 测试工作流
