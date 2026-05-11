import numpy as np
import torch

print("Debugging test case...")
test_x1 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
test_x2 = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32)
test_silu = torch.nn.functional.silu(test_x1)
test_y = test_silu * test_x2

print("test_x1:", test_x1)
print("test_silu:", test_silu)
print("test_y:", test_y)

expected = np.array([
    -2.0 * (1.0 / (1.0 + np.exp(2.0))),
    -2.0 * (1.0 / (1.0 + np.exp(1.0))),
    0.0,
    2.0 * (1.0 / (1.0 + np.exp(-1.0))),
    2.0 * (1.0 / (1.0 + np.exp(-2.0)))
], dtype=np.float32)

print("\nexpected:", expected)
diff = np.abs(test_y.numpy() - expected)
print("\ndiff:", diff)
print("max_diff:", np.max(diff))
