import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.array(X, dtype=float)
    if mode == "test":
        return [[round(float(v), 4) for v in row] for row in X]
    mask = np.array(mask, dtype=float)
    if drop_prob == 0:
        return [[round(float(v), 4) for v in row] for row in X]
    out = X*mask/(1-drop_prob)
    return [[round(float(v), 4) for v in row] for row in out]