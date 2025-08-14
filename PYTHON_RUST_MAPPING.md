# Python-Rust API 对应关系文档

本文档详细说明了 Python API 和 Rust 实现之间的对应关系，确保两者保持同步和兼容性。

## 文件结构对应

### Python 文件
```
src/llama/
├── __init__.py              # Python 包入口
├── model.py                 # ModelArgs 和 Transformer 类
├── generation.py            # Llama 高级接口
├── tokenizer.py            # 分词器接口
└── test_tokenizer.py       # 分词器测试
```

### Rust 文件
```
src/llama/
├── lib.rs                  # Rust 模块入口
├── model/
│   ├── transformer.rs      # 对应 Python Transformer
│   ├── model.rs           # 原始模型实现
│   ├── attention.rs       # 注意力机制
│   ├── feedforward.rs     # 前馈网络
│   └── generation.rs      # 生成算法
├── init/
│   ├── config.rs          # 对应 Python ModelArgs
│   └── config_new.rs      # 新的配置实现
├── ptensor/               # 张量操作
├── kernel/                # 计算内核
├── memory/                # 内存管理
└── compiler/              # 操作编译
```

## 类和结构体对应

### 1. 配置类对应

| Python | Rust | 描述 |
|--------|------|------|
| `ModelArgs` | `Config` | 模型配置参数 |

#### 字段对应关系
```python
# Python ModelArgs
@dataclass
class ModelArgs:
    hidden_size: int = 4096
    vocab_size: int = 32000
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    # ... 其他字段
```

```rust
// Rust Config
#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    // ... 其他字段
}
```

### 2. 模型类对应

| Python | Rust | 描述 |
|--------|------|------|
| `Transformer` | `Transformer<T>` | 主要模型结构 |

#### 方法对应关系
```python
# Python Transformer
class Transformer:
    def __init__(self, model_args: ModelArgs)
    def forward(self, sequences: List[List[int]]) -> List[List[float]]
    def generate(self, prompt_tokens: List[List[int]], ...) -> List[List[int]]
    def load_state_dict(self, state_dict: Dict, strict: bool = True)
```

```rust
// Rust Transformer
impl<T> Transformer<T> {
    pub fn new(config: Config, ...) -> Self
    pub fn forward(&self, sequences: *mut usize) -> Tensor<T>
    pub fn generate(&self, prompt_tokens: &[Vec<usize>], ...) -> Vec<Vec<usize>>
    // load_state_dict 通过 Cache 系统实现
}
```

### 3. 高级接口对应

| Python | Rust | 描述 |
|--------|------|------|
| `Llama` | `LlamaWrapper` (需实现) | 高级生成接口 |

## 数据类型映射

### 基础类型
| Python | Rust | 说明 |
|--------|------|------|
| `int` | `usize` | 非负整数 |
| `float` | `f32` | 单精度浮点 |
| `List[int]` | `Vec<usize>` | 整数向量 |
| `List[List[int]]` | `Vec<Vec<usize>>` | 二维整数向量 |
| `Optional[T]` | `Option<T>` | 可选值 |
| `Dict[str, Any]` | `HashMap<String, Value>` | 字典/映射 |

### 张量类型
| Python | Rust | 说明 |
|--------|------|------|
| 模拟的 logits | `Tensor<T>` | 实际的张量实现 |
| NumPy arrays | `Tensor<f16>` 或 `Tensor<f32>` | 权重数据 |

## API 接口对应

### 1. 模型创建
```python
# Python
model_args = ModelArgs.from_config_file("config.json")
model = Transformer(model_args)
```

```rust
// Rust
let mut config = Config::new();
config.load_model_config("config.json")?;
let model = create_transformer_from_config(config, cpu_num);
```

### 2. 权重加载
```python
# Python
state_dict = {...}  # 权重字典
model.load_state_dict(state_dict)
```

```rust
// Rust
// 通过 Cache 系统加载权重
let cache = Rc::new(RefCell::new(Cache::new(weights)));
// 权重在创建 Transformer 时传入
```

### 3. 前向传播
```python
# Python
sequences = [[1, 2, 3, 4]]
logits = model.forward(sequences)
```

```rust
// Rust
let mut sequences = vec![1, 2, 3, 4, 0, 0, ...];  // 填充到固定长度
let output = model.forward(sequences.as_mut_ptr());
```

### 4. 文本生成
```python
# Python
generated = model.generate(
    prompt_tokens=[[1, 2, 3]],
    max_gen_len=10,
    temperature=0.7,
    top_p=0.9
)
```

```rust
// Rust
let generated = model.generate(
    &[vec![1, 2, 3]],
    10,      // max_gen_len
    0.7,     // temperature
    0.9      // top_p
);
```

## 错误处理对应

### Python 异常
```python
try:
    model_args = ModelArgs.from_config_file("missing.json")
except FileNotFoundError:
    print("配置文件未找到")
except ValueError as e:
    print(f"配置错误: {e}")
```

### Rust Result
```rust
match config.load_model_config("missing.json") {
    Ok(_) => println!("配置加载成功"),
    Err(e) => println!("配置加载失败: {}", e),
}
```

## 性能特性对比

| 特性 | Python | Rust |
|------|--------|------|
| 类型安全 | 运行时检查 | 编译时检查 |
| 内存管理 | 垃圾回收 | 零成本抽象 |
| 并发性 | GIL 限制 | 原生并发 |
| 性能 | 解释执行 | 原生编译 |
| FFI 开销 | N/A | 最小化调用 |

## 数据流向

### 完整流程
```
Python Input -> JSON Serialization -> Rust FFI -> Rust Processing -> 
Rust Output -> JSON Serialization -> Python Output
```

### 具体示例
```
Python: [1, 2, 3] -> 
FFI: Vec<usize> -> 
Rust: Tensor<f16> -> 
Processing: forward() -> 
Rust: Tensor<f16> -> 
FFI: Vec<f32> -> 
Python: [[0.1, 0.2, ...]]
```

## 测试对应

### Python 测试
```python
def test_model_creation():
    args = ModelArgs(hidden_size=512, vocab_size=1000)
    model = Transformer(args)
    assert model.model_args.hidden_size == 512
```

### Rust 测试
```rust
#[test]
fn test_model_creation() {
    let config = Config {
        hidden_size: 512,
        vocab_size: 1000,
        ..Default::default()
    };
    let model = create_transformer_from_config(config, 1);
    assert_eq!(model.get_config().hidden_size, 512);
}
```

## 同步维护指南

### 1. 添加新配置参数
1. 在 Rust `Config` 中添加字段
2. 在 Python `ModelArgs` 中添加对应字段
3. 更新序列化/反序列化逻辑
4. 添加相应测试

### 2. 添加新方法
1. 在 Rust 中实现核心逻辑
2. 在 Python 中添加 FFI 包装
3. 确保接口签名一致
4. 添加文档和测试

### 3. 修改现有功能
1. 同时修改 Python 和 Rust 版本
2. 确保向后兼容性
3. 更新测试用例
4. 更新文档

## 性能优化策略

### Python 端优化
- 减少 FFI 调用次数
- 批量传输数据
- 使用类型提示提高可读性

### Rust 端优化
- 使用泛型减少代码重复
- 实现零拷贝数据传输
- 优化内存布局

### 接口优化
- 设计高效的数据传输格式
- 减少不必要的数据转换
- 实现流式处理接口

## 调试指南

### Python 调试
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
model = Transformer(model_args)
model._debug = True
```

### Rust 调试
```rust
// 使用 log crate
log::debug!("Forward pass with {} sequences", batch_size);

// 条件编译调试代码
#[cfg(debug_assertions)]
println!("Debug: tensor shape {:?}", tensor.shape);
```

这个对应关系文档确保了 Python 和 Rust 实现保持一致，为开发者提供了清晰的参考。
