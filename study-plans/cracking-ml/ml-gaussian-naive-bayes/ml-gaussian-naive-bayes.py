
import numpy as np

def gaussian_nb(X_train, y_train, X_test):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    n = len(y_train)
    classes = np.unique(y_train)
    eps = 1e-9 
    stats = {}
    for c in classes:
        mask = y_train == c 
        Xc = X_train[mask]
        stats[c] = {
            'prior': np.log(np.sum(mask) / n),
            'mean': np.mean(Xc, axis=0),
            'var': np.var(Xc, axis=0) + eps
        }

    predictions = []
    for x in X_test:
        best_class = None 
        best_score = -np.inf 
        for c in classes:
            s = stats[c]
            log_post = s['prior']
            log_post += np.sum(
                -0.5 * np.log(2*np.pi*s['var'])
                -
                (x-s['mean'])**2/(2*s['var'])
            )
            if log_post > best_score:
                best_score = log_post 
                best_class = c
        predictions.append(int(best_class))
    print(predictions)
    return predictions
