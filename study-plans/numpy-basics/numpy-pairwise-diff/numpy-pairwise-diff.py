import numpy as np

def pairwise_diff(a: list) -> np.ndarray:
    """
    Returns an (n, n) float64 array of signed pairwise differences.
    """
    arr = np.array(a, dtype=np.float64)
    return arr[:,None]-arr[None,:]
