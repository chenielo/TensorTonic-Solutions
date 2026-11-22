import numpy as np

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.RandomState(seed)
    indices = rng.choice(X.shape[0], size=k, replace=False)
    centroids = X[indices].copy()
    D = np.zeros((X.shape[0], k))
    for _ in range(max_iters):
        for i in range(X.shape[0]):
            for j in range(k):
                diff = X[i] - centroids[j]
                D[i, j] = np.sum(diff**2)
    
        labels = np.argmin(D,axis=1)
    
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                centroids[j] = points.mean(axis=0)
    for i in range(X.shape[0]):
        for j in range(k):
            diff = X[i] - centroids[j]
            D[i, j] = np.sum(diff**2)

    labels = np.argmin(D,axis=1)            
    return [labels.tolist(), np.round(centroids, 4).tolist()]