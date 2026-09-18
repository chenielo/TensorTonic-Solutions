import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X,dtype=float)
    centered = X - np.mean(X, axis=0)
    cov = centered.T @ centered / (X.shape[0]-1)
    std = np.sqrt (np.diag(cov))
    denom = np.outer(std, std)
    return cov/denom