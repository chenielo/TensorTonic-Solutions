import numpy as np

def cart_regress(X_train, y_train, X_test, max_depth=5, min_samples=2):
    """
    Returns: list of predicted values rounded to 4 decimal places
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    def mse(y):
        n =len(y)
        if n == 0:
            return 0 
        return np.mean((y-np.mean(y))**2)

    def best_split(X,y):
        n, d = X.shape 
        best_reduc = -1.0 
        best_feat = None
        best_thres = None 
        parent_mse = mse(y)
        for feat in range(d):
            thresholds = np.unique(X[:, feat])
            for thres in thresholds:
                left_mask = X[:, feat] <= thres
                right_mask = ~left_mask 
                n_left = np.sum(left_mask)
                n_right = n-n_left
                if n_left ==0 or n_right ==0:
                    continue 
                reduction = parent_mse - (n_left/n)*(mse(y[left_mask])) - (n_right/n)*(mse(y[right_mask]))
                if reduction > best_reduc:
                    best_reduc = reduction
                    best_feat = feat 
                    best_thres = thres 
        return best_feat, best_thres, best_reduc 


    def build_tree(X, y, depth):
        if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) == 1:
            classes, counts = np.unique(y, return_counts = True)
            return {'leaf': True, 'label': float(np.mean(y))}
        feat, thres, gain = best_split(X,y)
        if feat is None or gain <= 0:
            classes, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'label': float(np.mean(y))}
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

    tree = build_tree(X_train, y_train, 0)
    return [round(pred_one(tree, x), 4) for x in X_test]

