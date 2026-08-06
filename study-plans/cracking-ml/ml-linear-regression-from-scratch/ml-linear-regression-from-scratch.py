import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0 
    for _ in range(epochs):
        y_hat = X @ w + b 
        e = y_hat - y
        dw = (2/n) * (X.T @ e)
        db = (2/n) * np.sum(e)
        w -= lr*dw 
        b -= lr*db 
    weight = [float(round(v, 4)) for v in w]
    bias = float(round(b, 4))
    return (weight, bias)
        
