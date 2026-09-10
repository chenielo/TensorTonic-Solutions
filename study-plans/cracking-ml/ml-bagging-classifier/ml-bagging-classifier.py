import numpy as np

def bagging_classify(X_train, y_train, X_test, n_estimators=10, max_depth=5, seed=42):
    """
    Returns: list of predicted class labels for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    
    def gini(y):
        n = len(y)
        if n == 0:
            return 0 
        impurity = 1
        for c in np.unique(y):
            p = np.sum(y==c)/n 
            impurity -= p*p
        return impurity 

    def best_split(X,y):
        n, d = X.shape 
        best_gain = -1.0 
        best_feat = None
        best_thres = None 
        parent_gini = gini(y)
        for feat in range(d):
            thresholds = np.unique(X[:, feat])
            for thres in thresholds:
                left_mask = X[:, feat] <= thres
                right_mask = ~left_mask 
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                if n_left ==0 or n_right ==0:
                    continue 
                gain = parent_gini - (n_left/n)*(gini(y[left_mask])) - (n_right/n)*(gini(y[right_mask]))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat 
                    best_thres = thres 
        return best_feat, best_thres, best_gain 


    def build_tree(X, y, depth):
        if depth >= max_depth or len(np.unique(y)) == 1:
            classes, counts = np.unique(y, return_counts = True)
            return {'leaf': True, 'label': classes[np.argmax(counts)]}
        feat, thres, gain = best_split(X,y)
        if feat is None or gain <= 0:
            classes, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'label': classes[np.argmax(counts)]}
        left_mask = X[:,feat] <= thres
        return {
            'leaf': False, 'feature': feat, 'threshold': thres,
            'left': build_tree(X[left_mask], y[left_mask], depth+1),
            'right': build_tree(X[~left_mask], y[~left_mask], depth+1)
        }

    def pred_one(node, x):
        if node['leaf']:
            return node['label']
        if x[node['feature']] <= node['threshold']:
            return pred_one(node['left'],x)
        return pred_one(node['right'], x)

    trees = []
    for _ in range(n_estimators):
        idx = rng.randint(0, n, size=n)
        tree = build_tree(X_train[idx], y_train[idx], 0)
        trees.append(tree)

    preds = []
    for x in X_test:
        votes = [pred_one(tree, x) for tree in trees]
        unique_votes, counts = np.unique(votes, return_counts=True) 
        preds.append(int(unique_votes[np.argmax(counts)]))
    return preds

