import numpy as np

def permutation_importance(X, y, predict_fn, n_repeats=5, seed=42):
    """
    Returns: list of importance scores (one per feature) rounded to 4 decimal places
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    n, d = X.shape

    baseline = np.mean(predict_fn(X)==y)
    importances = np.zeros(d)

    for feat in range(d):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, feat] = rng.permutation(X_perm[:, feat])
            score = np.mean(predict_fn(X_perm)==y)
            scores.append(baseline-score)
        importances[feat] = np.mean(scores)

    return [round(float(x),4) for x in importances]