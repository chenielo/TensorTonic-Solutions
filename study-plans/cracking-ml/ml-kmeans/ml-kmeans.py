import numpy as np
from sklearn.cluster import KMeans
def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """
    # model = KMeans(
    #     n_clusters = k,
    #     init = "random",
    #     n_init = 1, 
    #     max_iter = max_iters,
    #     random_state = seed 
    # )
    # labels = model.fit_predict(X)
    # centroids = model.cluster_centers_ 
    X = np.asarray(X, dtype=float)
    rng = np.random.RandomState(seed)
    indices = rng.choice(X.shape[0], size=k, replace=False)
    centroids = X[indices].copy()
    for _ in range(max_iters):
        # (N, 1, D) - (1, K, D) -> (N, K, D)
        distance = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)        
        labels = np.argmin(distance, axis=1)
        new_centroids = centroids.copy()
        for cluster in range(k):
            points = X[labels == cluster]
            if len(points) > 0:
                new_centroids[cluster] = np.mean(points, axis=0)
            converged = np.array_equal(new_centroids, centroids)
            # if converged:
            #     centroids = new_centroids
            #     break 
            centroids = new_centroids
                
    distance = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
    labels = np.argmin(distance, axis=1)
    
    return [labels.tolist(), np.round(centroids, 4).tolist()]
        