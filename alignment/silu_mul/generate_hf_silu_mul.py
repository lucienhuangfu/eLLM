import torch
import numpy as np
from pathlib import Path

# Get script directory
script_dir = Path(__file__).parent
dump_dir = script_dir / "dump"

def generate_silu_mul_reference():
    """Generate reference outputs for fused SiLU + multiply operator."""
    print("===== Generating SiLU+Mul Reference =====")
    
    # Test configuration
    batch_size = 4
    hidden_size = 2048
    head_num = 32
    head_size = 128
    
    # Generate test inputs
    print(f"\nTest 1: Basic test with sequential values")
    x1 = torch.arange(batch_size * head_num * head_size, dtype=torch.float32).reshape(batch_size, head_num, head_size)
    x2 = torch.ones_like(x1)
    silu_out = torch.nn.functional.silu(x1)
    y = silu_out * x2
    np.save(dump_dir / "python_silu_mul_basic.npy", y.numpy())
    print(f"Saved to {dump_dir / 'python_silu_mul_basic.npy'}")
    print(f"Shape: {y.shape}")
    
    print(f"\nTest 2: All zeros")
    x1_zero = torch.zeros(batch_size, head_num, head_size, dtype=torch.float32)
    x2_zero = torch.ones_like(x1_zero)
    silu_out_zero = torch.nn.functional.silu(x1_zero)
    y_zero = silu_out_zero * x2_zero
    np.save(dump_dir / "python_silu_mul_zeros.npy", y_zero.numpy())
    print(f"Saved to {dump_dir / 'python_silu_mul_zeros.npy'}")
    
    print(f"\nTest 3: Small random values")
    torch.manual_seed(0)
    x1_rand = torch.randn(batch_size, head_num, head_size, dtype=torch.float32) * 0.01
    x2_rand = torch.randn_like(x1_rand) * 0.01
    silu_out_rand = torch.nn.functional.silu(x1_rand)
    y_rand = silu_out_rand * x2_rand
    np.save(dump_dir / "python_silu_mul_rand.npy", y_rand.numpy())
    print(f"Saved to {dump_dir / 'python_silu_mul_rand.npy'}")
    
    print(f"\nTest 4: Extreme values")
    x1_extreme = torch.tensor([-100, -10, -1, 0, 1, 10, 100], dtype=torch.float32).reshape(1, 7, 1)
    x2_extreme = torch.ones_like(x1_extreme)
    silu_out_extreme = torch.nn.functional.silu(x1_extreme)
    y_extreme = silu_out_extreme * x2_extreme
    np.save(dump_dir / "python_silu_mul_extreme.npy", y_extreme.numpy())
    print(f"Saved to {dump_dir / 'python_silu_mul_extreme.npy'}")
    print(f"Shape: {y_extreme.shape}")

if __name__ == "__main__":
    generate_silu_mul_reference()
