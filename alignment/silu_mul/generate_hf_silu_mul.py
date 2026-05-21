import torch
import numpy as np
from pathlib import Path

def generate_silu_mul_reference():
    """
    Generate SiLU multiply reference values using PyTorch (Hugging Face style).
    """
    # Use canonical test shapes
    batch_size = 4
    hidden_size = 2048

    # Generate test inputs with fixed seed for reproducibility
    torch.manual_seed(0)
    x1 = torch.randn((batch_size, hidden_size), dtype=torch.float32) * 0.1
    x2 = torch.randn((batch_size, hidden_size), dtype=torch.float32) * 0.1

    # Compute reference output (SiLU of x1 multiplied with x2) using PyTorch
    silu_out = torch.nn.functional.silu(x1)
    y = silu_out * x2

    return x1.numpy(), x2.numpy(), y.numpy()


def main():
    print("===== SiluMul =====");

    x1, x2, y = generate_silu_mul_reference();

    dump_dir = Path(__file__).parent / "dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(str(dump_dir / "python_silu_mul_x1.npy"), x1)
    np.save(str(dump_dir / "python_silu_mul_x2.npy"), x2)
    np.save(str(dump_dir / "python_silu_mul_output.npy"), y)

    print(f"x1 shape: {x1.shape}")
    print(f"x2 shape: {x2.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Saved to {dump_dir}/")


if __name__ == "__main__":
    main()
