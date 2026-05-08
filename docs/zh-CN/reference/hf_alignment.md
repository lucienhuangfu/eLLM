# Hugging Face 对齐参考

本页说明仓库如何使用一个 Hugging Face 风格的 Python 参考实现，
对齐 Rust 侧的 transformer 行为。

## 目标

参考代码不属于运行时，只用于生成小规模 golden case，
让 Rust 和 Python 可以按层对比。

推荐流程：

1. 在 `tests/reference/hf/cases/*.json` 里描述最小 case
2. 用 Python 参考实现生成 golden 文件
3. 在 Rust 测试里读取同一份 golden 文件并做比对

## 当前 RoPE 参考

当前 Rust 实现在 [`src/transformer/rope.rs`](../../../src/transformer/rope.rs)。
Python 参考实现位于 [`tests/reference/hf/rope.py`](../../../tests/reference/hf/rope.py)，
它和 Rust 保持同样的行为：

- 基础 RoPE 频率生成
- 部分 rotary 通道的尾部保持为 `1, 0`
- YaRN scaling 的解析与 attention scaling

## 输出格式

参考脚本输出一个 JSON 对象，包含：

- `head_dim`
- `rotary_dim`
- `max_sequence_length`
- `theta`
- `attention_scaling`
- `values`

`values` 的展平顺序与 Rust 保持一致：
每个 position 依次输出交错的 `cos, sin` 对，再追加 identity tail
通道的 `1, 0`。

## 生成 golden 文件

Windows PowerShell 下可以这样运行：

```powershell
.\tests\reference\hf\run_rope_reference.ps1
```

也可以直接调用：

```powershell
py -3 .\tests\reference\hf\rope.py --case .\tests\reference\hf\cases\rope_case_min.json --output .\tests\reference\hf\golden\rope_case_min.json
```

