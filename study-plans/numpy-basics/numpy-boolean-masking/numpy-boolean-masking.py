import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    arr = np.array(data, dtype=np.float64)
    elem_mask = (arr > threshold).astype(np.float64)
    any_mask = np.any(arr > threshold, axis = 1)
    all_mask = np.all(arr > threshold, axis = 1)
    any_filtered = np.where(any_mask[:, np.newaxis], arr, 0.0)
    all_filtered = np.where(all_mask[:, np.newaxis], arr, 0.0)
    return np.stack([elem_mask, any_filtered, all_filtered])