import numpy as np

def averaged_perceptron(X_train, y_train, X_test, n_epochs=10):
    """
    Returns: A list of predicted labels (-1 or +1) for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    n, d = X_train.shape 
    w = np.zeros(d)
    b = 0.0
    w_sum = np.zeros(d)
    b_sum = 0.0 
    count = 0

    for e in range(n_epochs):
        for i in range(n):
            if y_train[i] * (w@X_train[i]+b)<=0:
                w = w + y_train[i]*X_train[i]
                b = b + y_train[i]
            w_sum += w 
            b_sum += b 
            count += 1
    w_avg = w_sum/count 
    b_avg = b_sum / count 
    predictions = []
    for x in X_test:
        score = w_avg @ x + b_avg 
        predictions.append(1 if score > 0 else -1)
    return predictions 
    
