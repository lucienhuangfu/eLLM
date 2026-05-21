import numpy as np
from pathlib import Path


def compare_arrays(a, b, name, verbose=True):
    """
    Compare two numpy arrays and return whether they meet acceptance criteria.
    
    Args:
        a: First array (Python/HF reference)
        b: Second array (Rust implementation)
        name: Name of the test case
        verbose: Whether to print results
    
    Returns:
        bool: True if all criteria are met
    """
    if verbose:
        print(f"\n===== {name} =====")
        print(f"a shape: {a.shape}")
        print(f"b shape: {b.shape}")

    if a.shape != b.shape:
        if verbose:
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

    if verbose:
        print(f"\nmax_abs: {max_abs:.2e}")
        print(f"mean_abs: {mean_abs:.2e}")
        print(f"cosine: {cosine:.8f}")

    passed = max_abs < 1e-5 and mean_abs < 1e-6 and cosine > 0.999999
    
    if verbose:
        print(f"\n{'PASS' if passed else 'FAIL'}")

    if not passed and verbose and len(a) < 100:
        print("\nFirst 10 elements:")
        for i in range(min(10, len(a))):
            print(f"  [{i}] Python={a[i]:.8f}, Rust={b[i]:.8f}, diff={abs(a[i]-b[i]):.2e}")

    return passed


def load_npy(filepath):
    """
    Load numpy array from file.
    
    Args:
        filepath: Path to .npy file
    
    Returns:
        numpy array
    """
    return np.load(str(filepath))


def save_npy(filepath, data):
    """
    Save numpy array to file.
    
    Args:
        filepath: Path to save .npy file
        data: Numpy array to save
    """
    np.save(str(filepath), data)


def get_dump_dir(script_file):
    """
    Get dump directory path relative to a script file.
    
    Args:
        script_file: __file__ of the calling script
    
    Returns:
        Path object pointing to dump directory
    """
    return Path(script_file).parent / "dump"
