# Python API for Rust Transformer

This document describes the Python API that interfaces with the Rust-based Transformer implementation.

## Overview

The Python API provides a familiar interface for using the high-performance Rust implementation of the Llama transformer model. The API is designed to be compatible with common PyTorch-style usage patterns while leveraging the performance benefits of the Rust backend.

## Core Components

### ModelArgs Class

The `ModelArgs` class corresponds to the `Config` struct in Rust and contains all model configuration parameters.

```python
from llama.model import ModelArgs

# Create with default values
model_args = ModelArgs()

# Create with custom values
model_args = ModelArgs(
    hidden_size=4096,
    vocab_size=32000,
    num_hidden_layers=32,
    num_attention_heads=32,
    max_seq_len=2048
)

# Load from config.json file
model_args = ModelArgs.from_config_file("path/to/config.json")
```

#### Key Parameters

- `hidden_size`: Hidden dimension size (default: 4096)
- `vocab_size`: Vocabulary size (default: 32000) 
- `num_hidden_layers`: Number of transformer layers (default: 32)
- `num_attention_heads`: Number of attention heads (default: 32)
- `num_key_value_heads`: Number of key-value heads for GQA (default: same as attention heads)
- `max_seq_len`: Maximum sequence length (default: 2048)
- `rms_norm_eps`: RMS normalization epsilon (default: 1e-6)

### Transformer Class

The `Transformer` class provides the main model interface that calls into the Rust backend.

```python
from llama.model import Transformer

# Create transformer
model = Transformer(model_args)

# Forward pass
sequences = [[1, 2, 3, 4, 5]]  # Token sequences
logits = model.forward(sequences)

# Generation
generated = model.generate(
    prompt_tokens=sequences,
    max_gen_len=10,
    temperature=0.7,
    top_p=0.9
)
```

#### Methods

- `forward(sequences, start_pos=0)`: Run forward pass through transformer
- `generate(prompt_tokens, max_gen_len, temperature, top_p)`: Generate text
- `load_state_dict(state_dict, strict=True)`: Load model weights
- `eval()`: Set to evaluation mode (always enabled in Rust)

### Llama Class

The `Llama` class provides a high-level interface similar to the original Llama implementation.

```python
from llama.generation import Llama

# Build model from directory
llama = Llama.build(
    ckpt_dir="path/to/model",
    tokenizer_path="path/to/tokenizer.model", 
    max_seq_len=2048,
    max_batch_size=1
)

# Generate text
prompt_tokens = [[1, 10, 20, 30]]
generated_tokens, logprobs = llama.generate(
    prompt_tokens=prompt_tokens,
    max_gen_len=20,
    temperature=0.6,
    top_p=0.9,
    logprobs=True
)
```

## Rust Integration

### Data Flow

1. **Python Input** → **Rust Processing** → **Python Output**
2. Python provides familiar API surface
3. Rust handles all heavy computation
4. Results converted back to Python format

### Key Benefits

- **Performance**: Rust implementation with optimized kernels
- **Memory Efficiency**: f16 precision and efficient memory layout  
- **Safety**: Rust's memory safety guarantees
- **Compatibility**: PyTorch-style API for easy adoption

### Backend Mapping

| Python Component | Rust Component |
|------------------|----------------|
| `ModelArgs` | `Config` struct |
| `Transformer` | `Transformer<T>` struct |
| `forward()` | `Transformer::forward()` |
| `generate()` | Generation algorithms |
| Weight loading | SafeTensors loader |

## Usage Examples

### Basic Model Creation

```python
from llama.model import ModelArgs, Transformer

# Create model configuration
args = ModelArgs(hidden_size=2048, num_hidden_layers=16)

# Create transformer
model = Transformer(args)

# Use the model
sequences = [[1, 2, 3]]
output = model.forward(sequences)
```

### Loading from Config File

```python
from llama.model import create_transformer_from_config

# Load model from config.json
model = create_transformer_from_config(
    "models/llama-7b/config.json",
    max_seq_len=512  # Override config value
)
```

### High-level Generation

```python
from llama.generation import Llama

# Build complete model
llama = Llama.build(
    ckpt_dir="models/llama-7b",
    tokenizer_path="tokenizer.model",
    max_seq_len=2048,
    max_batch_size=1
)

# Generate text
tokens, logprobs = llama.generate(
    prompt_tokens=[[1, 10, 20]],
    max_gen_len=50,
    temperature=0.7
)
```

## Error Handling

The API includes proper error handling for common issues:

- Missing configuration files
- Invalid model parameters  
- Tokenizer loading errors
- Memory allocation failures

```python
try:
    model_args = ModelArgs.from_config_file("config.json")
    model = Transformer(model_args)
except FileNotFoundError:
    print("Config file not found")
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Performance Considerations

- **Batch Size**: Larger batches improve throughput
- **Sequence Length**: Longer sequences use more memory
- **Precision**: f16 reduces memory usage vs f32
- **CPU Cores**: Rust backend uses all available CPU cores

## Future Enhancements

Planned improvements to the Python API:

1. **GPU Support**: CUDA backend integration
2. **Streaming**: Token-by-token generation
3. **Quantization**: INT8/INT4 model support  
4. **Model Parallelism**: Multi-device inference
5. **Custom Kernels**: User-defined operations

## Migration from PyTorch

For users migrating from PyTorch implementations:

```python
# PyTorch style (familiar API)
model = Transformer(model_args)
output = model.forward(input_ids)

# But powered by Rust backend for performance
# No changes needed to user code
```

The API maintains compatibility while providing significant performance improvements through the Rust implementation.
