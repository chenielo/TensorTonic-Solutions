import numpy as np
from collections import Counter 
def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    predictions = []
    for x in X_test:
        dists = np.sqrt(np.sum((X_train - x)**2,axis=1))
        nearest_idx = np.argsort(dists)[:k]
        nearest_labels = y_train[nearest_idx]
        counts = Counter(nearest_labels.tolist())
        max_count = max(counts.values())
        best_label = min(label for label, c in counts.items() if c == max_count) 
        predictions.append(best_label)
    return predictions
        