import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    """
    Returns one predicted label for each test row.
    """
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test).tolist()
    # X_train = np.asarray(X_train, dtype=float)
    # y_train = np.asarray(y_train, dtype=int)
    # X_test = np.asarray(X_test, dtype=float)
    # predictions =[]
    # for point in X_test:

    #     order = sorted(range(len(X_train)), key=lambda index: (float(np.sum((X_train[index] - point) ** 2)), index))        
    #     counts = {}
    #     for index in order[:k]:
    #         label = y_train[index]
    #         counts[label] = counts.get(label, 0)+1
    #     predictions.append(int(min(counts, key=lambda label: (-counts[label], label))))
    return predictions