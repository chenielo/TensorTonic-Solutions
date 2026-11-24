import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    y = np.asarray(y, dtype=float)
    w = [0.0]*d
    b = 0.0
    for _ in range(epochs):
        y_hat = X@w+b 
        error = y_hat - y
        w -= lr*(2/len(y))*X.T@error
        b -= lr*(2/len(y))*np.sum(error)
    return [w, b]