import numpy as np

def outer_sum(a: list, b: list) -> np.ndarray:
    """
    Returns an (m, n) float64 array of pairwise sums.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a[:, None] + b[None, :]
