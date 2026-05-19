# Operator Alignment Tests

This directory contains operator alignment tests that verify the correctness of our Rust implementations against reference implementations.

## Setup

Install Python dependencies:
```bash
pip install -r alignment/requirements.txt
```

## Structure

```
alignment/
├── README.md
├── requirements.txt
├── alignment_utils.py          # Shared utilities for comparison
├── scripts/
│   └── create_new_operator.py  # Template tool for adding new operators
├── rope/
│   ├── dump/
│   │   ├── python_rope_basic.npy
│   │   ├── python_rope_partial.npy
│   │   ├── python_rope_yarn.npy
│   │   ├── rust_rope_basic.bin
│   │   ├── rust_rope_partial.bin
│   │   └── rust_rope_yarn.bin
│   ├── generate_hf_rope.py
│   ├── rope_alignment.py
│   ├── rope_alignment_test.rs
│   └── test_rope_alignment.py
└── [future_operator]/
    ├── dump/
    ├── [reference files]
    └── [test scripts]
```

## How to Add a New Operator

### Quick Start (Using Template Tool)

Use the template tool to create a new operator alignment test:

```bash
python alignment/scripts/create_new_operator.py <operator_name>
```

Example:
```bash
python alignment/scripts/create_new_operator.py linear
```

This will automatically create:
- `alignment/<operator_name>/` directory
- `dump/` subdirectory
- `generate_hf_<operator>.py` - Reference implementation template
- `test_<operator>_alignment.py` - Test script template
- `<operator>_alignment_test.rs` - Rust test template

### Manual Setup

1. Create a new directory under `alignment/` for your operator
2. Add a `dump/` subdirectory for test outputs
3. Implement the reference implementation (usually in Python)
4. Write the comparison/test script
5. Test and verify

## RoPE (Rotary Position Embedding)

**Location:** `alignment/rope/`

To run RoPE alignment tests:
```bash
python alignment/rope/test_rope_alignment.py
```

**Status:** ✅ All tests pass

## Utilities

### alignment_utils.py

Shared utility functions for all alignment tests:
- `compare_arrays(a, b, name)` - Compare two arrays with standard metrics
- `load_npy(filepath)` - Load numpy array from file
- `save_npy(filepath, data)` - Save numpy array to file
- `get_dump_dir(script_file)` - Get dump directory path
