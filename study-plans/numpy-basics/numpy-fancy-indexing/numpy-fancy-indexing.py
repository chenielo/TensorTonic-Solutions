import numpy as np

def select_by_index(arr: list, indices: list, axis: int) -> np.ndarray:
    """
    Returns a 2D float64 array of the selected rows or columns.
    """
    arr = np.asarray(arr, dtype=float)
    return arr[:, indices] if axis == 1 else arr[indices, :]
