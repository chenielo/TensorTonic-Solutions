import numpy as np

def sort_with_indices(data: list, axis: int) -> np.ndarray:
    """
    Returns a (2, m, n) float64 array of sorted values and source indices.
    """
    a = np.array(data, dtype = np.float64)
    sorted_vals = np.sort(a, axis=axis)
    sort_idx = np.argsort(a, axis=axis).astype(np.float64)
    return np.asarray([sorted_vals, sort_idx])
