import numpy as np

def scale_cols(data, weights):
    """Returns: np.ndarray of shape (m, n), each column scaled by corresponding weight"""
    arr = np.array(data, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    return arr*w