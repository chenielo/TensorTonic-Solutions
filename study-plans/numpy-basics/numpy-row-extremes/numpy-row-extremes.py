import numpy as np

def row_extremes(data: list) -> np.ndarray:
    """
    Returns float64 rows of maxima, maximum indices, minima, and minimum indices.
    """
    a  = np.array(data,dtype=np.float64)
    m = a.shape[0]
    row_idx = np.arange(m)
    max_col = np.argmax(a,axis=1)
    min_col = np.argmin(a,axis=1)
    max_val = a[row_idx, max_col]
    min_val = a[row_idx, min_col]
    return np.array([max_val, max_col.astype(np.float64),min_val,min_col.astype(np.float64)])
