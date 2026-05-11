import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path to import alignment_utils
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from alignment_utils import compare_arrays

def main():
    print("===== SiLU+Mul Alignment Test =====")
    
    dump_dir = script_dir / "dump"
    
    # First generate Python reference outputs
    print("\n--- Generating Python reference outputs ---")
    batch_size = 4
    hidden_size = 2048
    head_num = 32
    head_size = 128
    
    # Test 1: Basic
    x1 = torch.arange(batch_size * head_num * head_size, dtype=torch.float32).reshape(batch_size, head_num, head_size)
    x2 = torch.ones_like(x1)
    silu_out = torch.nn.functional.silu(x1)
    y_basic = silu_out * x2
    np.save(dump_dir / "python_silu_mul_basic.npy", y_basic.numpy())
    
    # Test 2: Zeros
    x1_zero = torch.zeros(batch_size, head_num, head_size, dtype=torch.float32)
    x2_zero = torch.ones_like(x1_zero)
    silu_out_zero = torch.nn.functional.silu(x1_zero)
    y_zero = silu_out_zero * x2_zero
    np.save(dump_dir / "python_silu_mul_zeros.npy", y_zero.numpy())
    
    # Test 3: Random
    torch.manual_seed(0)
    x1_rand = torch.randn(batch_size, head_num, head_size, dtype=torch.float32) * 0.01
    x2_rand = torch.randn_like(x1_rand) * 0.01
    silu_out_rand = torch.nn.functional.silu(x1_rand)
    y_rand = silu_out_rand * x2_rand
    np.save(dump_dir / "python_silu_mul_rand.npy", y_rand.numpy())
    
    # Try to load Rust outputs
    try:
        rust_basic = np.load(dump_dir / "rust_silu_mul_basic.npy")
        rust_zeros = np.load(dump_dir / "rust_silu_mul_zeros.npy")
        rust_rand = np.load(dump_dir / "rust_silu_mul_rand.npy")
        
        print("\n--- Comparing outputs ---")
        all_passed = True
        all_passed = compare_arrays(y_basic.numpy(), rust_basic, "SiLU+Mul (Basic)") and all_passed
        all_passed = compare_arrays(y_zero.numpy(), rust_zeros, "SiLU+Mul (Zeros)") and all_passed
        all_passed = compare_arrays(y_rand.numpy(), rust_rand, "SiLU+Mul (Random)") and all_passed
        
        print(f"\n===== {'All tests PASSED' if all_passed else 'Some tests FAILED'} =====")
    except FileNotFoundError as e:
        print(f"\nRust output file not found: {e}")
        print("\nTo run full alignment:")
        print("1. Create a Rust binary to export the fused operator's output")
        print("2. Save outputs as .npy files in the dump directory")
        print("\nFor now, verifying Python reference logic...")
        
        # Verify Python logic is correct
        print("\n--- Verifying Python logic ---")
        test_x1 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        test_x2 = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32)
        test_silu = torch.nn.functional.silu(test_x1)
        test_y = test_silu * test_x2
        
        # Calculate expected using NumPy (same math, different implementation)
        def silu_np(x):
            return x / (1.0 + np.exp(-x))
        expected_x1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected_silu = silu_np(expected_x1)
        expected = expected_silu * np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        
        diff = np.abs(test_y.numpy() - expected)
        max_diff = np.max(diff)
        if max_diff < 1e-6:
            print("[OK] Python reference logic verified")
        else:
            print(f"[ERROR] Python reference logic error, max diff: {max_diff}")

if __name__ == "__main__":
    main()
