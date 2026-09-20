import numpy as np

def normalize(data: list) -> np.ndarray:
    """
    Returns a float64 matrix standardized independently by column.
    """
    a = np.array(data, dtype=np.float64)
    mu = np.mean(a, axis=0)
    sigma = np.std(a, axis=0)
    return (a-mu)/sigma