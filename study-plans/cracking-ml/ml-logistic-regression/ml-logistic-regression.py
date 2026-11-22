import numpy as np
from sklearn.linear_model import LogisticRegression
 

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    # model = LogisticRegression(max_iter=n_iters)
    # model.fit(X, y)
    # w = model.coef_[0]
    # b = model.intercept_[0]
    
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(n_iters):
        z = X @ w + b 
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = p-y 
        w -= lr * X.T @ error / len(y) 
        b -= lr * np.mean(error)
    return [round(float(value), 4) for value in w], round(float(b), 4)
    
