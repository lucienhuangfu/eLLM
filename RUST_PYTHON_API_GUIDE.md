# eLLM Rust Python API 指南

本文档详细介绍如何为 Rust 编写 Python API，以及如何使用 eLLM 项目中的实现。

## 概述

eLLM 项目采用 **PyO3 + maturin** 方案来为 Rust 代码创建 Python 接口，提供了以下优势：

- 🚀 **高性能**: Rust 后端提供接近原生 C++ 的性能
- 🐍 **Python 友好**: 熟悉的 PyTorch 风格 API
- 🔄 **零拷贝**: 在 Python 和 Rust 之间高效传递数据
- 🛡️ **内存安全**: Rust 的内存安全保证

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Python 用户 API                          │
├─────────────────────────────────────────────────────────────┤
│  ellm.ModelArgs  │  ellm.Transformer  │  ellm.Llama        │
├─────────────────────────────────────────────────────────────┤
│                   Python 包装层                            │
├─────────────────────────────────────────────────────────────┤
│    PyO3 绑定层 (python_bindings/)                          │
├─────────────────────────────────────────────────────────────┤
│                 Rust 核心实现                               │
│  Config  │  Transformer  │  Attention  │  FFN              │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 Python 依赖
pip install maturin numpy
```

### 2. 构建扩展

```bash
# 自动构建和安装
python build.py

# 或手动构建
maturin develop --features python
```

### 3. 基本使用

```python
from ellm import ModelArgs, Transformer

# 创建模型配置
config = ModelArgs(
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    vocab_size=32000
)

# 创建模型
model = Transformer(config)

# 加载权重（需要实际权重文件）
model.load_state_dict(state_dict)

# 前向传播
sequences = [[1, 2, 3, 4, 5]]
logits = model.forward(sequences)

# 文本生成
generated = model.generate(sequences, max_gen_len=50)
```

## API 参考

### ModelArgs 类

配置类，对应 Rust 的 `Config` 结构体：

```python
config = ModelArgs(
    hidden_size=4096,           # 隐藏层大小
    vocab_size=32000,           # 词汇表大小  
    num_hidden_layers=32,       # Transformer 层数
    num_attention_heads=32,     # 注意力头数
    max_seq_len=2048,          # 最大序列长度
    rms_norm_eps=1e-6          # RMS 归一化 epsilon
)

# 从配置文件加载
config = ModelArgs.from_config_file("config.json")
```

### Transformer 类

主要的模型类，包装 Rust 实现：

```python
model = Transformer(config)

# 加载权重
model.load_state_dict(weights_dict)

# 前向传播
logits = model.forward(
    sequences,      # List[List[int]] - 输入序列
    start_pos=0     # int - 起始位置
)

# 文本生成
generated = model.generate(
    prompt_tokens,   # List[List[int]] - 提示 tokens
    max_gen_len=100, # int - 最大生成长度
    temperature=0.6, # float - 采样温度
    top_p=0.9       # float - 核采样参数
)
```

### Llama 高级接口

简化的文本生成接口：

```python
llama = Llama(
    model_path="path/to/model",
    tokenizer_path="path/to/tokenizer"
)

# 简单生成
response = llama.generate("解释人工智能")

# 聊天对话
messages = [
    {"role": "user", "content": "你好"}
]
response = llama.chat_completion(messages)
```

## 实现详解

### 1. PyO3 绑定实现

在 `src/llama/rust/python_bindings/` 目录下实现：

#### config.rs - 配置绑定
```rust
#[pyclass(name = "Config")]
pub struct PyConfig {
    pub inner: Config,
}

#[pymethods]
impl PyConfig {
    #[new]
    fn new(hidden_size: usize, vocab_size: usize, ...) -> PyResult<Self>
    
    #[classmethod]
    fn from_json_file(_cls: &PyType, path: String) -> PyResult<Self>
    
    #[getter]
    fn hidden_size(&self) -> usize
}
```

#### transformer.rs - 模型绑定
```rust
#[pyclass(name = "Transformer")]
pub struct PyTransformer {
    config: Arc<Config>,
    weights_loaded: bool,
}

#[pymethods]
impl PyTransformer {
    fn forward(&self, input_ids: &PyList, start_pos: usize) -> PyResult<PyObject>
    fn generate(&self, prompt_tokens: &PyList, ...) -> PyResult<Vec<Vec<usize>>>
    fn load_state_dict(&mut self, state_dict: &PyDict) -> PyResult<()>
}
```

### 2. 数据类型转换

Python 和 Rust 之间的数据转换：

| Python 类型 | Rust 类型 | 转换方式 |
|-------------|-----------|----------|
| `List[int]` | `Vec<usize>` | PyList 提取 |
| `Dict[str, Any]` | `HashMap<String, T>` | PyDict 遍历 |
| `numpy.ndarray` | `Tensor<T>` | numpy crate |
| `str` | `String` | 直接提取 |

### 3. 错误处理

使用 PyO3 的错误处理机制：

```rust
fn risky_operation() -> PyResult<()> {
    if condition_failed {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Operation failed"
        ));
    }
    Ok(())
}
```

## 构建配置

### Cargo.toml 配置

```toml
[lib]
name = "ellm"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module", "abi3-py38"] }
numpy = "0.20"

[features]
default = []
python = ["pyo3", "numpy"]
```

### pyproject.toml 配置

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "python"
module-name = "ellm._lowlevel"
bindings = "pyo3"
features = ["pyo3/extension-module"]
```

## 性能优化

### 1. 零拷贝数据传递

```rust
use numpy::{PyArray1, PyArray2};

// 直接使用 numpy 数组的内存
fn process_array(arr: &PyArray2<f32>) -> PyResult<()> {
    let slice = unsafe { arr.as_slice()? };
    // 直接操作内存，无需拷贝
    Ok(())
}
```

### 2. 批处理优化

```rust
// 处理整个批次而不是逐个元素
fn batch_forward(sequences: Vec<Vec<usize>>) -> Vec<Vec<f32>> {
    // Rust 高效批处理实现
}
```

### 3. 并行计算

```rust
use rayon::prelude::*;

// 利用 Rust 的并行迭代器
sequences.par_iter()
    .map(|seq| process_sequence(seq))
    .collect()
```

## 调试指南

### 1. 编译错误

```bash
# 查看详细错误信息
cargo build --features python --verbose

# 检查 PyO3 版本兼容性
maturin develop --features python --verbose
```

### 2. 运行时错误

```python
# 启用 Rust 日志
import os
os.environ['RUST_LOG'] = 'debug'

# Python 端调试
import traceback
try:
    model.forward(sequences)
except Exception as e:
    traceback.print_exc()
```

### 3. 性能分析

```bash
# Rust 性能分析
cargo bench

# Python 性能分析
python -m cProfile examples.py
```

## 常见问题

### Q: 如何处理大型张量？

A: 使用 numpy 数组的零拷贝接口：

```rust
use numpy::PyArray2;

fn handle_large_tensor(py: Python, data: &PyArray2<f32>) -> PyResult<()> {
    let shape = data.shape();
    let slice = unsafe { data.as_slice()? };
    // 直接操作内存
    Ok(())
}
```

### Q: 如何实现异步操作？

A: 使用 PyO3-asyncio：

```rust
use pyo3_asyncio::tokio::future_into_py;

#[pymethods]
impl PyTransformer {
    fn async_generate<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        future_into_py(py, async move {
            // 异步生成逻辑
            Ok(())
        })
    }
}
```

### Q: 如何处理 GIL？

A: 在计算密集型操作中释放 GIL：

```rust
fn compute_intensive_task(py: Python) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        // 释放 GIL 进行计算
        expensive_rust_computation()
    })
}
```

## 进阶主题

### 1. 自定义张量类型

```rust
#[pyclass]
struct RustTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[pymethods]
impl RustTensor {
    fn numpy_array(&self, py: Python) -> PyResult<PyObject> {
        // 转换为 numpy 数组
    }
}
```

### 2. 内存池管理

```rust
struct TensorPool {
    pool: Vec<Vec<f32>>,
}

impl TensorPool {
    fn get_tensor(&mut self, size: usize) -> Vec<f32> {
        // 复用内存池中的张量
    }
}
```

### 3. CUDA 支持

```rust
#[cfg(feature = "cuda")]
mod cuda_ops {
    // CUDA 操作实现
}
```

## 总结

通过 PyO3 + maturin 方案，我们成功地为 Rust 创建了高性能的 Python API。这种方案提供了：

1. **高性能**: 充分利用 Rust 的性能优势
2. **易用性**: Python 风格的友好接口
3. **内存安全**: Rust 的内存安全保证
4. **可扩展性**: 易于添加新功能和优化

这为构建高性能的 AI 推理引擎提供了理想的解决方案。
