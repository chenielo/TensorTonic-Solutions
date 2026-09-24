import numpy as np

def summarize(data: list, axis: int) -> np.ndarray:
    """
    Returns float64 rows of mean, standard deviation, minimum, and maximum.
    """
    a = np.array(data, dtype=np.float64)
    return np.array([np.mean(a, axis=axis), np.std(a, axis=axis), np.min(a, axis=axis), np.max(a, axis=axis)])
