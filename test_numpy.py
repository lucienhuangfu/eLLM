import numpy as np
from pathlib import Path

print("Testing NumPy...")

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

batch_size = 4
hidden_size = 2048
np.random.seed(0)
x1 = np.random.randn(batch_size, hidden_size).astype(np.float32) * 0.1
x2 = np.random.randn(batch_size, hidden_size).astype(np.float32) * 0.1

silu_out = silu(x1)
y = silu_out * x2

dump_dir = Path("alignment/silu_mul/dump")
dump_dir.mkdir(exist_ok=True)

np.save(dump_dir / "python_silu_mul_x1.npy", x1)
np.save(dump_dir / "python_silu_mul_x2.npy", x2)
np.save(dump_dir / "python_silu_mul_output.npy", y)

print("Generated reference files!")
print("x1 shape:", x1.shape)
print("First 3 elements of x1:", x1[0, :3])
