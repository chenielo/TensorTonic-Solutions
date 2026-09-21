import numpy as np

def norm_diff(a: list, b: list, lo: float, hi: float) -> np.ndarray:
    """
    Returns a float64 array of absolute normalized differences.
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    a_clip = np.clip(a, lo, hi)
    b_clip = np.clip(b, lo, hi)
    a_norm = (a_clip - lo) / (hi - lo)
    b_norm = (b_clip - lo) / (hi - lo)
    return np.abs(a_norm - b_norm)