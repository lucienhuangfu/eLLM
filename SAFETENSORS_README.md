# Llama3 8B SafeTensors 模型加载器

这个项目实现了用Rust加载Llama3 8B的safetensors模型格式的功能。

## 功能特性

- 🚀 支持单文件和多文件safetensors格式
- 💾 高效的内存映射加载，减少内存使用
- 🔄 支持F16、F32、BF16数据类型自动转换到std::f16
- ✅ 完整的模型验证和层检查
- 📊 详细的内存使用情况分析

## 依赖项

在`Cargo.toml`中添加了以下依赖：

```toml
safetensors = "0.4.1"
memmap2 = "0.9.4"
half = { version = "2.4.1", features = ["num-traits", "std"] }
```

## 主要组件

### 1. SafeTensorsModelLoader
单文件模型加载器，适用于单个safetensors文件的模型。

### 2. MultiFileSafeTensorsLoader  
多文件模型加载器，适用于分片的safetensors模型。

### 3. 便民函数
`load_llama3_from_safetensors()` - 自动检测并加载单文件或多文件模型。

## 使用方法

### 基本使用

```rust
use eLLM::llama::model_loader::load_llama3_from_safetensors;

// 加载模型
let (config, weights) = load_llama3_from_safetensors("path/to/llama3-8b-model")?;

println!("模型类型: {}", config.model_type);
println!("隐藏层大小: {}", config.hidden_size);
println!("加载的权重数量: {}", weights.len());
```

### 使用命令行工具

项目包含一个完整的命令行工具来演示模型加载：

```bash
# 编译项目
cargo build --release

# 运行safetensors加载器
cargo run --bin safetensors_loader -- /path/to/your/llama3-8b-model

# 或使用默认路径
cargo run --bin safetensors_loader
```

### 详细使用示例

```rust
use eLLM::llama::model_loader::SafeTensorsModelLoader;

// 创建加载器
let loader = SafeTensorsModelLoader::new("path/to/model")?;

// 分别加载配置和权重
let config = loader.load_config()?;
let weights = loader.load_weights_f16()?;

// 验证模型完整性
for i in 0..config.num_hidden_layers {
    let q_proj = format!("model.layers.{}.self_attn.q_proj.weight", i);
    if weights.contains_key(&q_proj) {
        println!("Layer {} Q projection found", i);
    }
}
```

## 模型目录结构

模型目录应包含以下文件：

```
model_directory/
├── config.json              # 模型配置文件
├── model.safetensors        # 单文件模型 (或)
├── model-00001-of-00001.safetensors  # 分片文件
├── model-00002-of-00001.safetensors
└── ...
```

## 支持的文件命名模式

加载器会自动查找以下命名模式的文件：
- `model.safetensors`
- `pytorch_model.safetensors`
- `model-00001-of-00001.safetensors`
- `model-*.safetensors` (分片模式)

## 数据类型转换

- **F16**: 直接加载为std::f16
- **F32**: 转换为std::f16 (使用as转换)
- **BF16**: 先转换为f32，再转换为std::f16

## 内存使用

- 使用内存映射(mmap)减少内存占用
- F16格式下，8B模型约占用16GB内存
- 支持大型模型的分片加载

## 模型验证

加载器会自动验证以下组件：
- 基础层：embedding, norm, lm_head
- Transformer层：attention和MLP组件
- 参数完整性检查

## 示例输出

```
Loading Llama3 8B model from: models/llama3-8b-instruct
Found model file: models/llama3-8b-instruct/model.safetensors
✅ Successfully loaded Llama3 8B model!

📊 Model Configuration:
  Model Type: llama
  Hidden Size: 4096
  Layers: 32
  Attention Heads: 32
  Key-Value Heads: 8
  Vocabulary Size: 128256
  Max Position Embeddings: 8192
  RMS Norm Epsilon: 0.00001

💾 Memory Usage:
  Total Parameters: 8.03B
  Memory Usage (f16): 16.06 GB
  Loaded Tensors: 291

🔍 Verifying Key Layers:
  ✅ model.embed_tokens.weight: 524550144 params
  ✅ model.norm.weight: 4096 params  
  ✅ lm_head.weight: 524550144 params
  ✅ Complete transformer layers: 32/32

📈 Large Tensors:
  • model.embed_tokens.weight: 524.6M params (1049.1 MB)
  • lm_head.weight: 524.6M params (1049.1 MB)
  • model.layers.0.mlp.up_proj.weight: 45.1M params (90.2 MB)
  • model.layers.0.mlp.gate_proj.weight: 45.1M params (90.2 MB)
  • model.layers.0.mlp.down_proj.weight: 45.1M params (90.2 MB)

✅ Model loading and verification completed!
```

## 错误处理

常见错误和解决方案：

1. **"config.json not found"**: 确保模型目录包含配置文件
2. **"No safetensors file found"**: 检查safetensors文件是否存在且命名正确
3. **"Unsupported tensor dtype"**: 当前支持F16/F32/BF16，其他格式需要扩展

## 性能优化

- 使用内存映射避免完整加载到内存
- 支持并行加载多个文件
- 延迟加载，只在需要时读取数据

## 扩展功能

可以基于这个加载器实现：
- 模型量化
- 动态批处理
- GPU加速推理
- 流式生成

## 注意事项

1. 需要启用nightly Rust特性 `#![feature(f16)]`
2. 确保有足够的内存加载8B模型
3. safetensors文件需要是有效的格式
4. 建议使用SSD存储以提高加载速度
