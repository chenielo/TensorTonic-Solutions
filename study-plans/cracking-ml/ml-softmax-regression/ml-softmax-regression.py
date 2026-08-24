import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n, d = X.shape
    K = n_classes 

    W = np.zeros((d, K))
    b = np.zeros(K)
    Y_oh = np.zeros((n, K))
    Y_oh[np.arange(n), y] = 1.0

    for _ in range(n_iters):
        Z = X@W + b 
        Z -= Z.max(axis=1, keepdims=True)
        exp_z = np.exp(Z)
        P = exp_z / exp_z.sum(axis=1, keepdims=True)

        diff = P-Y_oh 
        dW = (1.0 / n) * (X.T @ diff)
        db = (1.0 / n) * diff.sum(axis=0)

        W -= lr * dW
        b -= lr * db

    return (W.tolist(), b.tolist())