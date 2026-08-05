import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0 

    for _ in range(epochs):
        y_hat = X @ w + b 
        error = y_hat - y 
        dw = (2/n) * (X.T @ error)
        db = (2/n) * np.sum(error)
        w -= lr * dw
        b -= lr * db 

    weight = [v for v in w]
    bias = b 
    return (weight, bias)
