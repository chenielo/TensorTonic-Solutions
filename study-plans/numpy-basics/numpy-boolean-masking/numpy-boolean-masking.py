import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    data = np.asarray(data, dtype=float)
    d1 = (data > threshold).astype(int)
    d2 = np.where(np.any(data > threshold, axis=1)[:, None], data, 0.0)
    d3 = np.where(np.all(data > threshold, axis=1)[:, None], data, 0.0)
    return np.stack([d1, d2, d3]) 
