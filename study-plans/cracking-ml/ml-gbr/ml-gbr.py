import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

def gbr_predict(X_train: list, y_train: list, X_test: list, n_estimators: int = 10, max_depth: int = 3, learning_rate: float = 0.1, seed: int = 42) -> list:
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    def mse(values): return float(np.mean((values - np.mean(values)) ** 2))
    def build(X, values, depth):
        if depth >= max_depth or len(values) < 2 or len(np.unique(values)) == 1:
            return {"leaf": True, "value": float(np.mean(values))}
        parent=mse(values); best_gain,best_feature,best_threshold=0.0,None,None
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:,feature]):
                left=X[:,feature] <= threshold
                if not np.any(left) or np.all(left): continue
                gain=parent-np.mean(left)*mse(values[left])-np.mean(~left)*mse(values[~left])
                if gain > best_gain: best_gain,best_feature,best_threshold=gain,feature,threshold
        if best_feature is None: return {"leaf":True,"value":float(np.mean(values))}
        left=X[:,best_feature] <= best_threshold
        return {"leaf":False,"feature":best_feature,"threshold":best_threshold,"left":build(X[left],values[left],depth+1),"right":build(X[~left],values[~left],depth+1)}
    def predict_one(node,row):
        while not node["leaf"]: node=node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]
    initial=float(np.mean(y_train)); train_predictions=np.full(len(y_train),initial); trees=[]
    for _ in range(n_estimators):
        tree=build(X_train,y_train-train_predictions,0); trees.append(tree)
        train_predictions += learning_rate*np.asarray([predict_one(tree,row) for row in X_train])
    output=[]
    for row in X_test:
        value=initial+learning_rate*sum(predict_one(tree,row) for tree in trees)
        output.append(round(float(value),4))
    # reg = GradientBoostingRegressor(
    #     n_estimators=100,
    #     learning_rate=0.1,
    #     max_depth=3,
    #     loss="squared_error",
    #     random_state=0,
    # )
    # reg.fit(X_train, y_train)
    # preds = reg.predict(X_test)

    return output