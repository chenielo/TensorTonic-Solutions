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

    weight = [float(round(v,4)) for v in w]
    bias = float(round(b, 4)) 
    return (weight, bias)
