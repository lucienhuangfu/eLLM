import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from alignment_utils import compare_arrays, load_npy
import generate_hf_silu_mul

def main():
    print("===== SiluMul Alignment Test =====")

    # Step 1: Generate Python reference data
    print("\n--- Generating Python reference data ---")
    x1, x2, python_output = generate_hf_silu_mul.generate_silu_mul_reference()
    
    dump_dir = Path(__file__).parent / "dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    np.save(dump_dir / "python_silu_mul_x1.npy", x1)
    np.save(dump_dir / "python_silu_mul_x2.npy", x2)
    np.save(dump_dir / "python_silu_mul_output.npy", python_output)
    print(f"Python data saved to {dump_dir}")
    
    # Step 2: Load and compare with Rust output
    try:
        rust_output = load_npy(dump_dir / "rust_silu_mul_output.npy")
    except FileNotFoundError:
        print("\n--- --- ---")
        print("Rust output not found yet!")
        print("Please run the following command first:")
        print("cargo run --bin silu_mul_alignment_test")
        print("--- --- ---")
        return
        
    all_passed = compare_arrays(python_output, rust_output, "SiluMul Output")
    print(f"\n{'All tests PASSED' if all_passed else 'Some tests FAILED'}")


if __name__ == "__main__":
    main()
