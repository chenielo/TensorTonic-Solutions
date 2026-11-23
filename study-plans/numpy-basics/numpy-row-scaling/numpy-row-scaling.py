import numpy as np

def scale_rows(data: list, weights: list) -> np.ndarray:
    """
    Returns a float64 matrix with each row multiplied by its weight.
    """
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return data*weights[:,None]
