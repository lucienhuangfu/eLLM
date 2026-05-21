#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_new_operator.py <operator_name>")
        print("Example: python create_new_operator.py linear")
        sys.exit(1)

    operator_name = sys.argv[1].lower()
    operator_class = operator_name.capitalize()

    script_dir = Path(__file__).parent
    alignment_dir = script_dir.parent
    operator_dir = alignment_dir / operator_name
    dump_dir = operator_dir / "dump"

    # Create directories
    operator_dir.mkdir(exist_ok=True)
    dump_dir.mkdir(exist_ok=True)
    
    # Create .gitkeep file in dump directory
    (dump_dir / ".gitkeep").touch(exist_ok=True)

    print(f"Creating alignment test for: {operator_name}")

    # Create generate_hf_{operator}.py
    generate_file = operator_dir / f"generate_hf_{operator_name}.py"
    generate_content = f'''import torch
import numpy as np
from pathlib import Path


def generate_{operator_name}_reference():
    """
    Generate {operator_class} reference values.
    Replace this with actual {operator_class} implementation.
    """
    # Example: Generate dummy data
    # Replace this with actual logic
    hidden_size = 512
    batch_size = 4

    x = torch.randn((batch_size, hidden_size), dtype=torch.float32)
    w = torch.randn((hidden_size, hidden_size), dtype=torch.float32)
    
    # Example: matrix multiplication (replace with actual operator)
    y = x @ w.T

    return x.numpy(), w.numpy(), y.numpy()


def main():
    print(f"===== {operator_class} =====")
    
    x, w, y = generate_{operator_name}_reference()
    
    np.save(str(Path(__file__).parent / "dump" / "python_{operator_name}_input.npy"), x)
    np.save(str(Path(__file__).parent / "dump" / "python_{operator_name}_weight.npy"), w)
    np.save(str(Path(__file__).parent / "dump" / "python_{operator_name}_output.npy"), y)
    
    print(f"Input shape: {{x.shape}}")
    print(f"Weight shape: {{w.shape}}")
    print(f"Output shape: {{y.shape}}")
    print(f"Saved to {operator_name}/dump/")


if __name__ == "__main__":
    main()
'''
    generate_file.write_text(generate_content)

    # Create test_{operator}_alignment.py
    test_file = operator_dir / f"test_{operator_name}_alignment.py"
    test_content = f'''import numpy as np
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
DUMP_DIR = SCRIPT_DIR / "dump"


def compare_arrays(a, b, name):
    print(f"\\n===== {{name}} =====")
    print(f"a shape: {{a.shape}}")
    print(f"b shape: {{b.shape}}")

    if a.shape != b.shape:
        print(f"FAIL: shape mismatch")
        return False

    diff = np.abs(a - b)
    max_abs = diff.max()
    mean_abs = diff.mean()

    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    cosine = np.dot(a_flat, b_flat) / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    print(f"\\nmax_abs: {{max_abs:.2e}}")
    print(f"mean_abs: {{mean_abs:.2e}}")
    print(f"cosine: {{cosine:.8f}}")

    passed = max_abs < 1e-5 and mean_abs < 1e-6 and cosine > 0.999999
    print(f"\\n{{'PASS' if passed else 'FAIL'}}")

    if not passed and len(a) < 100:
        print("\\nFirst 10 elements:")
        for i in range(min(10, len(a))):
            print(f"  [{{i}}] Python={{a[i]:.8f}}, Rust={{b[i]:.8f}}, diff={{abs(a[i]-b[i]):.2e}}")

    return passed


def main():
    print(f"===== {operator_class} Alignment Test =====")

    try:
        python_output = np.load(str(DUMP_DIR / f"python_{operator_name}_output.npy"))
        rust_output = np.load(str(DUMP_DIR / f"rust_{operator_name}_output.npy"))
    except FileNotFoundError as e:
        print(f"\\nError: {{e}}")
        print(f"\\nPlease run:")
        print(f"  1. python alignment/{operator_name}/generate_hf_{operator_name}.py")
        print(f"  2. Generate Rust output and save to {operator_name}/dump/rust_{operator_name}_output.npy")
        return

    all_passed = compare_arrays(python_output, rust_output, f"{operator_class} Output")

    print(f"\\n{{'All tests PASSED' if all_passed else 'Some tests FAILED'}}")


if __name__ == "__main__":
    main()
'''
    test_file.write_text(test_content)

    # Create Rust test template
    rust_file = operator_dir / f"{operator_name}_alignment_test.rs"
    rust_content = f'''// === alignment/{operator_name}/{operator_name}_alignment_test.rs ===
use std::fs;


fn write_npy(path: &str, data: &[f32], shape: &[usize]) {{
    let descr = npy::to_numpy_file(path, data, shape).unwrap();
}}


fn main() {{
    println!("===== {operator_class} Alignment Test =====");

    // TODO: Replace with actual Rust operator call
    // Example (replace with your operator):
    // let input = load_input();
    // let output = your_operator(input);
    // write_npy("alignment/{operator_name}/dump/rust_{operator_name}_output.npy", &output, &[shape]);

    println!("\\n--- Done ---");
}}
'''
    rust_file.write_text(rust_content)

    print(f"\n[OK] Successfully created alignment test for {operator_name}!")
    print(f"\nDirectory structure created at: {operator_dir}")
    print(f"\nNext steps:")
    print(f"  1. Edit {operator_name}/generate_hf_{operator_name}.py with reference implementation")
    print(f"  2. Edit {operator_name}/{operator_name}_alignment_test.rs with Rust implementation")
    print(f"  3. Run Python reference: python alignment/{operator_name}/generate_hf_{operator_name}.py")
    print(f"  4. Run Rust binary to generate output")
    print(f"  5. Run test: python alignment/{operator_name}/test_{operator_name}_alignment.py")


if __name__ == "__main__":
    main()
