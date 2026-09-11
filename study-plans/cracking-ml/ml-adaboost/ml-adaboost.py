import numpy as np

def adaboost_classify(X_train, y_train, X_test, n_estimators=10, seed=42):
    """
    Returns: list of predicted labels in {-1, +1} for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    n, d = X_train.shape

    def fit_slump(X,y,w):
        best_err = float('inf')
        best_feat = 0
        best_thres = 0.0
        best_polarity = 1
        for feat in range(d):
            for thres in np.unique(X[:, feat]):
                for polarity in [1, -1]:
                    preds = np.ones(n)
                    if polarity == 1:
                        preds[X[:, feat] <= thres] = -1
                    else:
                        preds[X[:, feat] > thres] = -1
                    err = np.sum(w[preds != y])
                    if err < best_err:
                        best_err = err 
                        best_feat = feat 
                        best_thres = thres
                        best_polarity = polarity
        return best_feat, best_thres, best_polarity, best_err

    def stump_pred(X, feat, thres, polarity):
        preds = np.ones(X.shape[0])
        if polarity == 1:
            preds[X[:, feat]<=thres] = -1
        else:
            preds[X[:, feat]>thres] = -1
        return preds 

    w = np.ones(n)/n 
    stumps=[]
    alphas=[]

    for _ in range(n_estimators):
        feat, thres, polarity, err = fit_slump(X_train, y_train, w)
        error = max(err, 1e-10)
        alpha = 0.5*np.log((1-err)/err)
        preds = stump_pred(X_train, feat, thres, polarity)
        w = w * np.exp(-alpha * y_train * preds)
        w = w/np.sum(w)
        stumps.append((feat, thres, polarity))
        alphas.append(alpha)

    result = []
    for x in X_test:
        score = 0.0
        for (feat, thres, polarity), alpha in zip(stumps, alphas):
            if polarity == 1:
                pred = -1 if x[feat]<= thres else 1
            else:
                pred = -1 if x[feat] > thres else 1
            score += alpha*pred 
        result.append(1 if score >= 0 else -1)
    return result