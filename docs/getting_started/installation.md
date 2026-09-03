# Installation

The current runnable service targets one model and one layout:

```text
models/Qwen3-Coder-30B-A3B-Instruct/
```

The directory must contain the complete Hugging Face snapshot, including the
tokenizer, configuration, generation configuration, and all safetensors shards.

## Requirements

- Linux on x86-64
- A recent Intel server CPU with AVX-512 FP16 support; AMX-FP16 is recommended
  for the optimized BRGEMM attention path
- Enough RAM for the approximately 61 GB model plus the configured graph and KV
  cache; 128 GiB or more is a practical starting point
- At least 80 GB of free disk space for the model and release build
- Rust installed with [rustup](https://rustup.rs/); this repository selects the
  required nightly toolchain through `rust-toolchain.toml`
- Python 3 and `curl` for model download and client examples

No GPU or NPU is required.

## Clone and download the model

```bash
git clone https://github.com/lucienhuangfu/eLLM.git
cd eLLM

python3 -m pip install --upgrade huggingface_hub
hf download Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --local-dir models/Qwen3-Coder-30B-A3B-Instruct
```

Some environments require a Hugging Face login before downloading the model:

```bash
hf auth login
```

## Build

```bash
cargo build --release --bin main --bin qwen3_coder_30b_a3b
```

The build uses `target-cpu=native`, so build the binary on the machine where it
will run. Do not copy it to a machine with an older CPU instruction set.

## Optional BRGEMM dependency

The default attention backend is `brgemm`. It dynamically loads
`libtorch_cpu.so` and automatically falls back to the native implementation if
the required library or CPU support is unavailable.

If PyTorch is installed in a normal Python location, eLLM discovers the library
automatically:

```bash
python3 -m pip install torch
```

For a non-standard installation, set its exact path:

```bash
export ELLM_LIBTORCH_CPU_PATH=/path/to/libtorch_cpu.so
```

Continue with the [Quickstart](quickstart.md).
